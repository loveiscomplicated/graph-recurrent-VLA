"""Independent readback of the native-resolution benchmark, without model runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statistics(values):
    a = np.asarray(values, float)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), minimum=float(a.min()), maximum=float(a.max()))


def main(root, report):
    protocol = read(root / "run/protocol.json")
    static = read(root / "run/static_raw.json")
    frames = read(root / "run/tracking_frames.json")
    sequences = read(root / "run/tracking_sequences.json")
    original_static = read(root / "run/static_summary.json")
    original_tracking = read(root / "run/tracking_summary.json")
    manifest = read(root / "run/input_manifest.json")
    capture = read(root / "run/capture_verification.json")
    completion = read(root / "completion.json")
    assert completion["status"] == "COMPLETE" and completion["CUDA_compute_processes_after"] == []
    identities = {stage: read(root / f"run/{stage}_identity.json") for stage in ("static", "tracking")}
    assert protocol["benchmark_source_sha256"] == digest(root / "benchmark.py")
    assert len(static) == 288 and len(frames) == 1512 and len(sequences) == 72
    assert len(manifest) == 528 and len(capture["audits"]) == 176
    assert capture["status"] == "PASS" and capture["physics_steps"] == 0
    assert all(a["physical_state_unchanged"] and a["calibration_agrees"] for a in capture["audits"])
    for stage, rows, csvname in (("static", static, "static_raw"), ("tracking", frames, "tracking_frames"),
                                 ("sequences", sequences, "tracking_sequences")):
        with (root / f"run/{csvname}.csv").open() as f:
            reread = list(csv.DictReader(f))
        assert len(reread) == len(rows)
        for a, b in zip(rows, reread):
            for key, value in a.items():
                assert b[key] == str(value), (stage, key)
    sizes = protocol["sizes"]
    paired = {}
    for size in sizes:
        s = [r for r in static if r["size"] == size]
        f = [r for r in frames if r["size"] == size and r["timestep"] > 0]
        seq = [r for r in sequences if r["size"] == size]
        assert len(s) == 96 and len(f) == 480 and len(seq) == 24
        assert {r["order_position"] for r in s} == {0, 1, 2}
        for pos in range(3):
            assert sum(r["order_position"] == pos for r in s) == 32
            assert sum(r["order_position"] == pos for r in seq) == 8
        for metric, value in original_static[str(size)].items():
            assert statistics([r[metric] for r in s]) == value
        for metric, value in original_tracking[str(size)]["steady"].items():
            assert statistics([r[metric] for r in f]) == value
        for metric, value in original_tracking[str(size)]["sequence"].items():
            assert statistics([r[metric] for r in seq]) == value
        for rows, keys in ((s, ("parent_id", "camera", "timestep")), (seq, ("parent_id", "camera"))):
            for key in {tuple(r[k] for k in keys) for r in rows}:
                group = [r for r in rows if tuple(r[k] for k in keys) == key]
                assert len({r["masks_sha256"] for r in group}) == 1
        paired[size] = {"static": s, "tracking": f}
    # Verify identical normalized prompts across resolutions, with no GT selection.
    for key in {(r["parent_id"], r["camera"], r["timestep"]) for r in manifest}:
        group = [r for r in manifest if (r["parent_id"], r["camera"], r["timestep"]) == key]
        assert len(group) == 3 and {r["size"] for r in group} == set(sizes)
        assert all(r["boxes_normalized"] == group[0]["boxes_normalized"] for r in group)
    # Cluster resampling keeps the four physical parents as units, rather than frames.
    parents = sorted({r["parent_id"] for r in static})
    delta = {}
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(parents), (10000, len(parents)))
    for mode in ("static", "tracking"):
        parent_means = {size: np.array([np.mean([r["total_wall_ms"] for r in paired[size][mode]
            if r["parent_id"] == p]) for p in parents]) for size in sizes}
        delta[mode] = {}
        for a, b in ((128, 512), (512, 768), (128, 768)):
            differences = parent_means[b]-parent_means[a]
            delta[mode][f"{a}_to_{b}"] = dict(parent_n=len(parents), mean_delta_ms=float(differences.mean()),
                mean_ratio=float(parent_means[b].mean()/parent_means[a].mean()),
                exploratory_parent_bootstrap_CI95_ms=np.quantile(differences[draws].mean(1), [.025, .975]).tolist())
    verification = dict(status="PASS", source_hash=True, raw_csv_json_exact=True, summaries_recomputed_exact=True,
        order_balanced=True, same_normalized_two_boxes=True, repeat_masks_exact=True, sizes=sizes,
        static_samples=288, tracking_total_frames=1512, tracking_steady_frames=1440, native_images=528,
        distinct_parents=4, distinct_tracking_sequences=8,
        checkpoint_identity=identities["static"]["checkpoint_sha256"] == identities["tracking"]["checkpoint_sha256"] == protocol["checkpoint_sha256"],
        run_ids={stage: identity["run_id"] for stage, identity in identities.items()},
        absent_static_scenes=sum(any(r["reference_absent"]) for r in manifest if r["size"] == 768 and r["timestep"] in (0,80)),
        local_execution="CPU JSON/CSV/numeric/plot readback only; CUDA and rendering on remote58576054")
    (root / "verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    (root / "paired_differences.json").write_text(json.dumps(delta, indent=2)+"\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, mode, title in zip(axes, ("static", "tracking"), ("Static: encode + two prompts", "Tracking: steady frame + CPU masks")):
        for parent in parents:
            values = [np.mean([r["total_wall_ms"] for r in paired[size][mode] if r["parent_id"] == parent]) for size in sizes]
            ax.plot(sizes, values, "o-", alpha=.55, linewidth=1, label=parent)
        means = [np.mean([r["total_wall_ms"] for r in paired[size][mode]]) for size in sizes]
        ax.plot(sizes, means, "o-", color="black", linewidth=2.2, label="Mean")
        ax.set(title=title, xlabel="Native square image width (pixels)", ylabel="Synchronized wall time (ms)", xticks=sizes)
        ax.set_ylim(0, 55)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("SAM2.1 Small / RTX3090 / internal1024 / two objects")
    fig.savefig(root / "resolution_latency.png", dpi=160)
    plt.close(fig)

    rel = "../../outputs/task10i/stage4a2/sam2_resolution_latency_20261103_v1"
    static_lines, tracking_lines = [], []
    for size in sizes:
        s = original_static[str(size)]
        f = original_tracking[str(size)]["steady"]["total_wall_ms"]
        seq = original_tracking[str(size)]["sequence"]
        static_lines.append(f'| {size}² | 96 | {s["encode_wall_ms"]["mean"]:.3f} | {s["two_prompt_output_wall_ms"]["mean"]:.3f} | {s["total_wall_ms"]["mean"]:.3f} | {s["total_wall_ms"]["median"]:.3f} | {s["total_wall_ms"]["p95"]:.3f} |')
        tracking_lines.append(f'| {size}² | 480 | {f["mean"]:.3f} | {f["median"]:.3f} | {f["p95"]:.3f} | {1000/f["mean"]:.2f} | {seq["initialization_wall_ms"]["mean"]:.3f} | {seq["peak_allocated_bytes"]["maximum"]/1024**3:.3f} |')
    ds = delta["static"]["128_to_768"]
    dt = delta["tracking"]["128_to_768"]
    text = f'''# Task 10-I — SAM2 입력 원본 해상도 128/512/768 소요 시간 비교

2026-11-03 KST. **동일 조건의 native 해상도 비교 완료.**
128→768에서 정적 분할 평균은 **{ds['mean_ratio']:.3f}배 ({ds['mean_delta_ms']:+.3f}ms)**,
연속 추적 평균은 **{dt['mean_ratio']:.3f}배 ({dt['mean_delta_ms']:+.3f}ms)**였다.
세 조건 모두 SAM2 내부 크기는1024²로 같았다. Native 픽셀 수 증가와 추론 시간 증가를 비례 관계로 해석하지 않는다.

## 실행 조건과 표본

- 기존 Vast.ai58576054, RTX3090 24GB. 실험 시작 시 다른 CUDA Compute Process0개.
- SAM2.1 Hiera Small, 동일 checkpoint SHA `{protocol['checkpoint_sha256']}`.
- FP32 Weight, CUDA BF16 autocast, TF32 허용, Batch1, CPU4 thread, compile 없음.
- Torch `{identities['static']['packages']['torch']}`, torchvision `{identities['static']['packages']['torchvision']}`, SAM-2 `{identities['static']['packages']['SAM-2']}`. 전체 Package/설치 SAM2 Source Hash는 Completion에 있다.
- 기존 네 Train Parent `{', '.join(parents)}`의 Main/Wrist를 모두 사용했다. 성적·가시성으로 대상을 거르지 않았다.
- 같은 seed의 geometry를 만들고 저장 qpos/qvel·Peg/Box pose를 복원했다. 세 해상도를 같은 scene/상태에서 직접 렌더링했다. 단순 이미지 확대가 아니다.
- 4 Parent×2 Camera×22시점(t0..20,t80)×3해상도=528 PNG. 카메라 pose/FOV/mount 동일, intrinsic 배율과 extrinsic 일치, capture 전후 물리 상태·TCP·geometry를 검사했다. Physics Step0, 새 Action/Policy Rollout0.
- Renderer/Simulator는 원격 CPU backend(`physx_cpu/sapien_cpu`)에서 실행했다. CUDA는 SAM2 추론에 사용했다. macOS에서 Simulator/MPS/CUDA는 실행하지 않았다.

## 프롬프트와 비교 통제

Detector/API 변동을 제거하기 위해 native768 Actor Segmentation의 bbox를 한 번 정규화하고 세 해상도에 같은 비율로 적용했다.
정적 분할은 각 프레임, 추적은 t0에만 두 Box를 사용했다. 미가시 Actor에는 full-image Box를 주고 **항상 두 객체를 실제 실행**했다.
정적16개 Camera-Frame 중 미가시 Actor 포함 장면은 {verification['absent_static_scenes']}개다. 이를 제외하거나 추론을 건너뛰지 않았다.
이는 **Oracle Box를 쓰는 시간 측정**이며 Detection/Mask 품질·일반화 평가가 아니다. GT bbox 생성 비용은 SAM2 시간에 포함하지 않았다.

정적 분할은16장×6회×3해상도=288회다. 각 장면에서 여섯 순열을 모두 실행해 순서 위치를 균형화했다.
해상도마다3회 warmup을 제외했다. 추적은8개 독립 Camera Sequence×3회×3해상도=72 Sequence,
1512 Frame이다. 세 순환 순서로 위치를 균형화했고 해상도마다21-frame warmup Sequence1개를 제외했다.
Steady 통계는 각 Sequence의 t1..20만 포함해 **해상도당480 Frame**이다. t0와 JPEG 전체 로딩·초기화는 별도다.
두 모드 모두 반복 간 같은 입력·같은 해상도의 최종 Mask Hash가 정확히 일치했다.

## 정적 분할: 한 이미지 인코딩 + 두 Box

단위 ms. RGB는 미리 메모리에 올렸다. Encode는 `set_image`의 전처리·H2D·인코딩,
Prompt/Output은 두 번의 순차 `predict`, 원본 크기 Mask 복원 및 NumPy 전송을 포함한다.
각 단계 전후 CUDA 동기화로 wall time을 측정했다. 모델 Load·PNG 읽기·Hash·저장은 제외했다.

| Native | n | Encode 평균 | 두 Prompt/Output 평균 | 전체 평균 | 전체 중앙값 | 전체 p95 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(static_lines)}

## 연속 추적: 프레임당 두 물체

단위 ms, VRAM은GiB. 각 조건에서 JPEG quality100/subsampling0을 사용했다. 원본 해상도에 따른 JPEG Decode 차이는 초기화에 포함한다.
`init_state(offload_video_to_cpu=True, offload_state_to_cpu=False, async_loading_frames=False)`로21개 Frame을 먼저 로드한다.
따라서 여기서 초기화는 **21-frame 파일 로딩+첫 인코딩+두 Box 입력**이며 온라인 최초1Frame latency가 아니다.
Steady는 `next(propagate_in_video)`부터 두 원본 크기 Bool Mask의 CPU 전송까지 포함한다. I/O·Hash·저장·Renderer는 제외했다.

| Native | Steady n | 평균 | 중앙값 | p95 | 1000/평균 FPS | 초기화 평균 | Peak allocated VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(tracking_lines)}

GPU 입력 Tensor는 정적 `[1,3,1024,1024]`, Video CPU Frame Buffer는 `[21,3,1024,1024]`로 모두 같다.
원본 크기 출력 복원·Bool 변환·CPU 전송과 입력 전처리는 해상도에 따라 달라질 수 있다.
추적은 동기화 wall time이며 GPU kernel만의 시간이나 Simulator/Detector/Graph/Controller 포함20Hz 검증이 아니다.
JPEG 경로와 무압축 RGB 정적 경로는 서로 다른 측정이므로 두 모드 사이의 시간 차이를 해상도 효과로 해석하지 않는다.

## 대응 차이와 한계

Parent별 평균의 대응 차이를 계산하고 Parent4개 단위 bootstrap10,000회(seed42)를 보조 기록했다.
반복·프레임을 독립 장면으로 세지 않았다. 작은4 Parent에서 나온 탐색적 구간이며 다른GPU/모델/긴Sequence 성능 보장은 아니다.

'''
    for mode, label in (("static", "정적"), ("tracking", "추적")):
        for change, d in delta[mode].items():
            lo, hi = d["exploratory_parent_bootstrap_CI95_ms"]
            text += f'- {label} {change.replace("_to_", "→")}: 평균 차이 {d["mean_delta_ms"]:+.3f}ms, 비율 {d["mean_ratio"]:.3f}배, 탐색적95% 구간 [{lo:+.3f}, {hi:+.3f}]ms.\n'
    text += f'''
모든 결과는 한 서버의 순차 실행 조건이다. 균형화로 순서 영향을 줄였지만 GPU clock/온도 변동을 제거했다고 주장하지 않는다.
Mask 정확도·3D 역투영 정밀도·삽입 성공률은 이번에 측정하지 않았다. 해상도 변경이 이 지표들을 개선한다는 결론은 별도 평가가 필요하다.

처음 정적 측정은 Package metadata에서 distribution 이름을 `sam2`로 조회해 모델 Load 전 실패했다.
실제 이름 `SAM-2`로 수정했고 원래 Source/Protocol·실패 Log를 보존했다. 시간 측정·샘플·모델 설정은 바꾸지 않았다.
중단된 Logger 상태를 실제 종료에 맞게 failed로 정리했고, Capture/정적/추적의 정상 Run3개는 모두 finished다.

## 검증과 산출물

공통 `utils.experiment_logger.ExperimentLogger`를 재사용했고 W&B는 disabled다.
정적 Run `{identities['static']['run_id']}`, 추적 Run `{identities['tracking']['run_id']}`.
Production/공통 Simulator API·Model Config·Checkpoint·기존 Stage4A-2 Runner/Stop Stub은 변경하지 않았다.
새 코드는 독립 `visual/experiments/sam2_resolution_latency_20261103_v1/`에만 추가했다. 외부 게시·학습·DINO/Qwen은 실행하지 않았다.

독립 CPU readback은 JSON/CSV 모든 값 일치, 표본 수, 균형화, 동일 정규화 Box, 반복 Mask Hash,
모든 요약 통계 재계산의 정확 일치, capture/calibration 검증을 확인했다. Python 구문 검사도 통과했다.
측정 종료 후 PNG/JPEG 전체1,032개·기존 Raw/GT·Checkpoint Hash를 재검사했고 CUDA Compute Process0개를 확인했다.
실행 원자료/환경/프로토콜46개 파일은 SHA256 일치를 확인해 로컬 Output에 복사했다.
PNG/JPEG1,032개는 원격 `/workspace/task10i_sam2_resolution_latency_20261103_v1/run/inputs/`에 보존돼 있다.
로컬에는 파일별 Hash와 전체 입력 Manifest가 있으며, 원본 이미지의 로컬 복사를 완료했다고 주장하지 않는다.

- [실행 프로토콜]({rel}/run/protocol.json) · [입력 manifest]({rel}/run/input_manifest.json) · [Capture 검증]({rel}/run/capture_verification.json)
- [정적 원자료 CSV]({rel}/run/static_raw.csv) · [추적 Frame CSV]({rel}/run/tracking_frames.csv) · [추적 Sequence CSV]({rel}/run/tracking_sequences.csv)
- [정적 요약]({rel}/run/static_summary.json) · [추적 요약]({rel}/run/tracking_summary.json) · [대응 차이]({rel}/paired_differences.json)
- [독립 readback 검증]({rel}/verification.json) · [실행 코드](../../visual/experiments/sam2_resolution_latency_20261103_v1/benchmark.py)
- [실행 완료·입력 보존·환경]({rel}/completion.json) · [아티팩트 SHA256]({rel}/artifact_sha256.json)

![Native 해상도별 Parent 평균 소요 시간]({rel}/resolution_latency.png)
'''
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text)
    print(json.dumps(dict(verification=verification, differences=delta), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("report", type=Path)
    args = p.parse_args()
    main(args.root, args.report)
