#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

namespace py = pybind11;
using at::Tensor;
using at::indexing::Slice;

namespace {

Tensor fit_plane(const Tensor& points, const Tensor& selected) {
    const auto weight = selected.to(points.scalar_type());
    const auto column = weight.unsqueeze(1);
    const auto center = (points * column).sum(0) / weight.sum().clamp_min(1);
    const auto centered = points - center;
    const auto covariance = centered.transpose(0, 1).matmul(centered * column);
    const auto vectors = std::get<1>(at::linalg_eigh(covariance, "L"));
    const auto normal = vectors.select(1, 0);
    return at::cat({normal, (-normal.matmul(center)).unsqueeze(0)});
}

Tensor fit_planes(
    const Tensor& points, int64_t minimum, int64_t trials, int64_t maximum,
    double tolerance, bool stop_when_stable, const py::function& sample_schedule
) {
    TORCH_CHECK(points.is_cuda() && points.scalar_type() == at::kFloat &&
                points.dim() == 2 && points.size(1) == 3,
                "CUDA RANSAC expects float32 CUDA points with shape [N, 3]");
    auto remaining = at::arange(points.size(0), points.options().dtype(at::kLong));
    std::vector<Tensor> result;
    for (int64_t iteration = 0; iteration < maximum; ++iteration) {
        const auto count = remaining.size(0);
        if (count < minimum) {
            break;
        }

        const auto current = points.index({remaining});
        const auto sample_count = std::min<int64_t>(count, 1300);
        Tensor schedule;
        {
            py::gil_scoped_acquire lock;
            schedule = sample_schedule(count, sample_count, trials).cast<Tensor>();
        }
        const auto sampled = current.index_select(0, schedule.to(points.device()));
        const auto sample_points = sampled.narrow(0, 0, sample_count);
        const auto triples = sampled.narrow(0, sample_count, 3 * trials).reshape({-1, 3, 3});
        const auto a = triples.select(1, 0);
        const auto b = triples.select(1, 1);
        const auto c = triples.select(1, 2);
        auto normal = at::linalg_cross(b - a, c - a);
        const auto lengths = at::linalg_vector_norm(normal, 2, {1});
        normal = normal / lengths.clamp_min(1e-7).unsqueeze(1);
        const auto offsets = -(normal * a).sum(1);
        const auto residual = (sample_points.matmul(normal.transpose(0, 1)) + offsets).abs();
        const auto votes = ((residual < tolerance) & (lengths > 1e-7).unsqueeze(0)).sum(0);
        const auto winner = votes.argmax().view({1});
        const auto winner_normal = normal.index_select(0, winner).select(0, 0);
        const auto winner_offset = offsets.index_select(0, winner).select(0, 0);
        auto selected = (current.matmul(winner_normal) + winner_offset).abs() < tolerance;
        auto sufficient = selected.sum() >= minimum;

        Tensor stable_plane;
        for (int refinement = 0; refinement < 3; ++refinement) {
            const auto previous = selected;
            const auto plane = fit_plane(current, selected);
            const auto plane_normal = plane.slice(0, 0, 3);
            const auto offset = plane.select(0, 3);
            selected = ((current.matmul(plane_normal) + offset).abs() < tolerance) & sufficient;
            if (refinement < 2) {
                sufficient = sufficient & (selected.sum() >= minimum);
            }
            if (stop_when_stable && at::equal(previous, selected)) {
                stable_plane = plane;
                break;
            }
        }

        auto remainder = remaining.index({at::bitwise_not(selected)});
        if (count - remainder.size(0) < minimum) {
            break;
        }
        if (!stable_plane.defined()) {
            stable_plane = fit_plane(current, selected);
        }
        result.push_back(stable_plane);
        remaining = remainder;
    }
    return result.empty() ? at::empty({0, 4}, points.options()) : at::stack(result);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("fit_planes", &fit_planes, py::call_guard<py::gil_scoped_release>());
}
