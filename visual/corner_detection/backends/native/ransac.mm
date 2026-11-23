#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <ATen/mps/MPSStream.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {

constexpr unsigned kGroupSize = 256;
constexpr unsigned kSampleLimit = 1300;
using Sample = int (*)(uint32_t, uint32_t, uint32_t, int32_t*, int32_t*);

struct Buffer {
    id<MTLBuffer> buffer;
    size_t offset = 0;
};

void bind(id<MTLComputeCommandEncoder> encoder, unsigned index, Buffer value) {
    [encoder setBuffer:value.buffer offset:value.offset atIndex:index];
}

template<class T>
void bind(id<MTLComputeCommandEncoder> encoder, unsigned index, const T& value) {
    [encoder setBytes:&value length:sizeof(T) atIndex:index];
}

template<class... Arguments>
void launch(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
            unsigned threads, unsigned group_size, const Arguments&... args) {
    [encoder setComputePipelineState:pipeline];
    unsigned index = 0;
    (bind(encoder, index++, args), ...);
    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

id<MTLComputePipelineState> pipeline(id<MTLLibrary> library, NSString* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) throw std::runtime_error([name UTF8String]);
    id<MTLComputePipelineState> result =
        [[library device] newComputePipelineStateWithFunction:function error:&error];
    if (!result) throw std::runtime_error([[error localizedDescription] UTF8String]);
    return result;
}

class Ransac {
public:
    explicit Ransac(const char* source) {
        device_ = at::mps::getCurrentMPSStream()->device();
        MTLCompileOptions* options = [MTLCompileOptions new];
        BOOL fast_math = NO;
        if (const char* flag = std::getenv("PYTORCH_MPS_FAST_MATH")) fast_math = *flag == '1';
        if (@available(macOS 26.0, *)) {
            options.languageVersion = MTLLanguageVersion4_0;
        } else if (@available(macOS 15.0, *)) {
            options.languageVersion = MTLLanguageVersion3_2;
        } else {
            options.languageVersion = MTLLanguageVersion3_1;
        }
        if (@available(macOS 15.0, *)) {
            options.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
            options.mathFloatingPointFunctions =
                fast_math ? MTLMathFloatingPointFunctionsFast : MTLMathFloatingPointFunctionsPrecise;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = fast_math;
#pragma clang diagnostic pop
        }
        NSError* error = nil;
        id<MTLLibrary> library = [device_ newLibraryWithSource:@(source) options:options error:&error];
        if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
        score_ = pipeline(library, @"score_planes_parallel");
        first_max_ = pipeline(library, @"native_first_max");
        select_ = pipeline(library, @"select_plane");
        fit_ = pipeline(library, @"weighted_plane");
        block_sizes_ = pipeline(library, @"native_outlier_block_sizes");
        compact_ = pipeline(library, @"native_compact_outliers");
        winner_ = allocate(sizeof(int64_t));
        inliers_ = allocate(sizeof(int32_t), MTLResourceStorageModeShared);
    }

    int fit(Buffer current, Buffer output, uint32_t size, int trials, int minimum,
            int maximum, float tolerance, Sample sampler) {
        auto* stream = at::mps::getCurrentMPSStream();
        reserve(size, trials);
        int accepted = 0;
        while (accepted < maximum && size >= static_cast<unsigned>(minimum)) {
            int samples = std::min(size, kSampleLimit);
            auto* schedule = static_cast<int32_t*>([schedule_ contents]);
            if (!sampler(size, samples, trials, schedule, schedule + samples)) {
                throw std::runtime_error("failed to generate the deterministic RANSAC schedule");
            }

            const bool need_remainder = accepted + 1 < maximum;
            const unsigned blocks = (size + kGroupSize - 1) / kGroupSize;
            Buffer plane{output.buffer, output.offset + 4 * sizeof(float) * accepted};
            Buffer remainder{scratch_[accepted % 2]};
            at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
                @autoreleasepool {
                    auto encoder = stream->commandEncoder();
                    // Earlier PyTorch operations on this stream produce the input cloud.
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    launch(encoder, score_, trials * 32, std::min(trials * 32u, kGroupSize),
                           current, Buffer{schedule_}, Buffer{schedule_, samples * sizeof(int32_t)},
                           Buffer{hypotheses_}, Buffer{votes_}, samples, tolerance);
                    launch(encoder, first_max_, 1, 1, Buffer{votes_}, Buffer{winner_}, trials);
                    launch(encoder, select_, size, std::min(size, kGroupSize),
                           current, Buffer{hypotheses_}, Buffer{winner_}, Buffer{selected_}, tolerance);
                    launch(encoder, fit_, kGroupSize, kGroupSize, current, Buffer{selected_},
                           plane, Buffer{inliers_}, size, tolerance, minimum, 3, 1);
                    if (need_remainder) {
                        launch(encoder, block_sizes_, blocks * kGroupSize, kGroupSize,
                               Buffer{selected_}, Buffer{sizes_}, size);
                        launch(encoder, compact_, blocks * kGroupSize, kGroupSize,
                               current, Buffer{selected_}, Buffer{sizes_}, remainder, size);
                    }
                    stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
                }
            });

            const int inliers = *static_cast<const int32_t*>([inliers_ contents]);
            if (inliers < minimum) break;
            if (static_cast<unsigned>(inliers) > size) {
                throw std::runtime_error("Metal RANSAC returned an invalid inlier count");
            }
            ++accepted;
            size -= inliers;
            current = remainder;
        }
        return accepted;
    }

private:
    id<MTLBuffer> allocate(size_t size, MTLResourceOptions options = MTLResourceStorageModePrivate) {
        id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:options];
        if (!buffer) throw std::runtime_error("Metal RANSAC buffer allocation failed");
        return buffer;
    }

    void reserve(uint32_t size, int trials) {
        if (size > point_capacity_) {
            const auto capacity = std::max(size, point_capacity_ * 2);
            auto selected = allocate(capacity);
            auto sizes = allocate((capacity + kGroupSize - 1) / kGroupSize * sizeof(uint32_t));
            auto first = allocate(capacity * 3 * sizeof(float));
            auto second = allocate(capacity * 3 * sizeof(float));
            selected_ = selected;
            sizes_ = sizes;
            scratch_[0] = first;
            scratch_[1] = second;
            point_capacity_ = capacity;
        }
        if (trials > trial_capacity_) {
            auto schedule = allocate((kSampleLimit + 3 * trials) * sizeof(int32_t),
                                     MTLResourceStorageModeShared);
            auto hypotheses = allocate(4 * trials * sizeof(float));
            auto votes = allocate(trials * sizeof(int32_t));
            schedule_ = schedule;
            hypotheses_ = hypotheses;
            votes_ = votes;
            trial_capacity_ = trials;
        }
    }

    id<MTLDevice> device_;
    id<MTLComputePipelineState> score_, first_max_, select_, fit_, block_sizes_, compact_;
    id<MTLBuffer> schedule_, hypotheses_, votes_, winner_, selected_, inliers_, sizes_;
    id<MTLBuffer> scratch_[2] = {};
    uint32_t point_capacity_ = 0;
    int trial_capacity_ = 0;
};

}  // namespace

extern "C" {
void* corner_ransac_create(const char* source, char* error, size_t error_size) {
    @autoreleasepool {
        try {
            return new Ransac(source);
        } catch (const std::exception& exception) {
            std::snprintf(error, error_size, "%s", exception.what());
            return nullptr;
        }
    }
}

void corner_ransac_destroy(void* instance) {
    delete static_cast<Ransac*>(instance);
}

int corner_ransac_fit(void* instance, void* points, size_t point_offset,
                      uint32_t size, void* planes, int trials, int minimum, int maximum,
                      float tolerance, Sample sampler, char* error, size_t error_size) {
    @autoreleasepool {
        try {
            return static_cast<Ransac*>(instance)->fit(
                {(__bridge id<MTLBuffer>)points, point_offset}, {(__bridge id<MTLBuffer>)planes},
                size, trials, minimum, maximum, tolerance, sampler);
        } catch (const std::exception& exception) {
            std::snprintf(error, error_size, "%s", exception.what());
            return -1;
        }
    }
}
}
