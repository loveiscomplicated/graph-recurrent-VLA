"""Read-only validation of restored training and completed reevaluation inputs."""
from __future__ import annotations

import json
from pathlib import Path

from recurrent.common import sha256
from recurrent.evaluate import evaluation_identity
from recurrent.integrity import evaluation_complete, read_eval_rows
from recurrent.reevaluate import prepare
from recurrent.run_experiments import eval_jobs


def load_reevaluation(cfg, source_root, evaluation_dir):
    """Validate archived identities using recorded evaluator versions, not plot-host versions.

    This never repairs JSONL or writes locks/metadata in either source tree.
    Source checkpoint/config/data contracts and all completion hashes remain strict.
    """
    source_root, evaluation_dir = Path(source_root).resolve(), Path(evaluation_dir).resolve()
    plan = json.loads((evaluation_dir / "plan.json").read_text())
    recorded = json.loads((evaluation_dir / "reevaluation.identity.json").read_text())
    expected, tasks, fixture = prepare(cfg, source_root, evaluation_dir, plan["num_envs"])
    if plan["identity"] != recorded:
        raise ValueError("Reevaluation plan identity mismatch")
    # Moving a verified archive and plotting under a different Python/Torch
    # installation do not change which environment performed the evaluation.
    ignored = {"source_root", "evaluation_versions"}
    if {k: v for k, v in expected.items() if k not in ignored} != {
        k: v for k, v in recorded.items() if k not in ignored
    }:
        raise ValueError("Reevaluation source/config identity mismatch")
    fixture_hash = sha256(fixture)
    import torch
    from recurrent.common import environment_record
    fixture_meta = torch.load(fixture, map_location="cpu", weights_only=False)
    ev = cfg["eval"]
    if (fixture_meta.get("source") != "legacy physx_cpu reset(seed)"
            or fixture_meta.get("environment") != environment_record(ev.get("horizon", 250))
            or fixture_meta.get("seeds") != list(range(ev["start_seed"], ev["start_seed"] + ev["episodes"]))
            or any(fixture_meta.get("versions", {}).get(k) != v for k, v in recorded["evaluation_versions"].items())):
        raise ValueError("Archived fixture metadata mismatch")
    rows, input_hashes, completion_hashes = [], {}, {}
    for task in tasks:
        target = Path(task["output"])
        identity = evaluation_identity(
            task["checkpoint"], eval_jobs(cfg, task["level"], task["name"] == "expert"),
            ev["episodes"], ev["start_seed"], ev.get("horizon", 250),
            cfg.get("corruption", "corruption.yaml"), task["backend"], None,
        )
        identity.update(initial_states=fixture_hash, versions=recorded["evaluation_versions"])
        if not evaluation_complete(target, identity):
            raise ValueError(f"Incomplete archived evaluation: {task['name']}")
        batch = read_eval_rows(target, identity)
        if len(batch) != task["expected_rows"] or any(
            (r["ckpt"], r["model"], r["train_level"], r["train_seed"])
            != (task["name"], task["model"], task["level"], task["train_seed"])
            for r in batch
        ):
            raise ValueError(f"Archived evaluation labels/count mismatch: {task['name']}")
        rows.extend(batch)
        input_hashes[task["name"]] = sha256(target)
        completion_hashes[task["name"]] = sha256(target.with_suffix(".complete.json"))
    complete = dict(
        source_identity_sha256=recorded["source_identity_sha256"], fixture_sha256=fixture_hash,
        policy_episodes=sum(t["expected_rows"] for t in tasks if t["name"] != "expert"),
        expert_episodes=sum(t["expected_rows"] for t in tasks if t["name"] == "expert"),
        completions=completion_hashes,
    )
    if json.loads((evaluation_dir / "reevaluation.complete.json").read_text()) != complete:
        raise ValueError("Root reevaluation completion mismatch")
    source_exp = source_root / cfg["out_dir"]
    histories = {t["name"]: sha256(source_exp / "runs" / t["name"] / "history.json")
                 for t in tasks if t["name"] != "expert"}
    provenance = dict(
        schema=1, training_runs=len(histories), policy_episodes=complete["policy_episodes"],
        expert_episodes=complete["expert_episodes"], source_identity_sha256=recorded["source_identity_sha256"],
        fixture_sha256=fixture_hash, source_versions=recorded["source_versions"],
        evaluation_versions=recorded["evaluation_versions"],
        checkpoint_sha256=recorded["checkpoint_sha256"], history_sha256=histories,
        evaluation_jsonl_sha256=input_hashes,
        root_completion_sha256=sha256(evaluation_dir / "reevaluation.complete.json"),
        policy_backend=recorded["policy_backend"], expert_backend=recorded["expert_backend"],
        rkn_covariance_update=recorded["rkn_covariance_update"],
    )
    receipt = source_root / "__backup_provenance__" / "restore-receipt.json"
    if receipt.exists():
        data = json.loads(receipt.read_text())
        provenance["training_archive"] = {k: data[k] for k in ("repo_id", "revision", "prefix")}
    return rows, source_exp, provenance


def shareable_preamble(cfg, idx, hist, provenance):
    """Describe the measured scope and limitations without inferring model causality."""
    from recurrent.report import group_rate

    models, levels = cfg["models"], cfg["levels"]
    leaders = []
    for level in levels:
        rates = [(group_rate(idx[(model, level, level, "normal")])["mean"], model) for model in models]
        best = max(rate for rate, _ in rates)
        leaders.append(f"- `{level}`: {', '.join(model for rate, model in rates if abs(rate-best)<1e-9)} "
                       f"{100*best:.1f}% (normal, 관측된 평균 성공률 기준)")
    source = provenance["source_versions"]
    evaluated = provenance["evaluation_versions"]
    lines = [
        "# Recurrent controller 84개 학습 실행과 전체 재평가", "",
        "7개 모델 × 손상 4단계 × 학습 seed 3개의 학습 결과와, 기존 best checkpoint를 사용한 "
        "정책 11,340 episode·CPU expert 60 episode를 정리했다. 새 학습이나 rollout 없이 완료된 기록에서 생성한 리포트다.", "",
        "## 핵심 결과", "",
        *leaders, "",
        "위 순위는 이 seed와 평가 조건에서 관측한 값이다. 모델 구조의 효과를 다른 학습 차이와 분리한 "
        "인과적 결론이나, 모집단에서의 확정적인 우열을 뜻하지 않는다.", "",
        "## 집계와 해석", "",
        "- 기본 표는 학습 손상과 평가 손상이 같은 조건이다. clean 학습의 교차평가와 high 학습의 blackout은 별도 집계한다.",
        "- 기본 표의 각 칸은 학습 seed 3개 × 동일한 평가 seed 30개다. 평균과 표본 표준편차(ddof=1)를 표시하고, "
        "`seed_success.png`에는 학습 seed별 성공률을 점으로 표시한다.",
        "- 동일한 평가 seed가 학습 seed 사이에서도 재사용된다. 기준 모델과의 비교는 학습 seed와 episode seed를 "
        "각각 재표집하는 crossed bootstrap 2,000회, 95% 구간이다. 학습 seed가 3개뿐이며 다중 비교 보정은 하지 않았다. "
        "탐색적 비교로 읽어야 한다.",
        "- `summary.csv`의 pooled Wilson 구간은 episode 합계에 대한 기술 통계이며 학습 seed 불확실성을 나타내지 않는다.",
        "- 종합 평균이 필요하면 `equal_weight_success.csv`의 손상 4단계 동일 가중 평균을 쓴다. "
        "교차평가·blackout까지 모든 episode를 합친 값으로 모델 순위를 계산하지 않는다.",
        "- Blackout은 high 학습 모델을 eval corruption none에서 평가한다. 안정적으로 잡고 들어 올린 경우에만 "
        "강제 차단되므로 실제 발생 수를 함께 표시한다. 발생한 episode만의 성공률은 선택된 하위집단의 기술 통계다.",
        "- Blackout 0에서도 성공률이 낮아 바닥 효과가 있다. 작은 변화나 비단조 곡선만으로 기억 능력의 우열을 단정하지 않는다.",
        "- 정책은 PhysX CUDA, expert는 PhysX CPU다. 같은 CPU 초기 fixture를 사용하지만 이후 물리·IK 경로는 다르다. "
        "expert는 CPU 참고 baseline이며 GPU 정책 성공률의 직접적인 상한으로 해석하지 않는다.",
        "- 산점도는 평가한 best.pt의 validation action MSE를 쓴다. 동률을 평균 순위로 처리한 Spearman 상관은 "
        "84개 실행 전체의 기술 통계이며, 모델·손상 조건이 섞인 상관을 인과 효과로 해석하지 않는다.",
        f"- 원본 학습 Torch `{source['torch']}`, 재평가 Torch `{evaluated['torch']}`. "
        "RKN은 기존 checkpoint 형식을 유지한 `stable_v1` 분산 갱신을 사용했다.",
        "- 실행 시 Git dirty 상태가 기록돼 있다. 원본 학습 시점의 미커밋 코드와 현재 코드가 동일하다고 보장하지 않는다. "
        "각 실행의 identity·checkpoint hash와 원본 archive를 출처로 보존한다.",
        "- 현재 서버에서 단일 정책 추론 지연을 새로 측정하지 않았다. GPU batch 추론 시간을 episode별 지연으로 나누어 제시하지 않는다.", "",
        "## 원자료와 재현", "",
    ]
    for label, key in [("원본 학습 백업", "training_archive"), ("전체 재평가 백업", "evaluation_archive")]:
        if key in provenance:
            archive = provenance[key]
            url = archive.get("url") or (f"https://huggingface.co/datasets/{archive['repo_id']}/tree/"
                                         f"{archive['revision']}/{archive['prefix']}")
            lines += [f"- [{label}]({url}) — revision `{archive['revision']}`"]
    lines += [
        "- HF 백업 열람에는 해당 private 저장소의 읽기 권한이 필요하다. 아래 PNG·CSV는 이 리포트에 포함돼 있다.",
        "- `provenance.json`: 84개 checkpoint·history, 85개 평가 JSONL, fixture·root 완료 기록의 SHA-256과 실행 패키지 버전.",
        "- 입력 검증은 원본 학습 완료, 평가 identity·episode 집합·summary·완료 hash를 모두 검사하며 원본을 수정하지 않는다.", "",
        "```bash",
        ".venv/bin/python -m recurrent.report \\",
        "  --config recurrent/configs/experiment.yaml \\",
        "  --source-root /workspace/recovered-main \\",
        "  --reevaluation-dir outputs/recurrent/reevaluation_main_20261023 \\",
        "  --evaluation-backup-receipt outputs/backups/reevaluation_main_20261023T163639Z/receipt.json \\",
        "  --out-dir docs/reports/recurrent_main_20261023",
        "```", "",
        "다른 위치에 복원했다면 source-root·reevaluation-dir 경로를 바꾸고, 로컬 게시 receipt가 없으면 "
        "evaluation-backup-receipt 옵션은 생략한다. 파일 생성만 수행하며 학습·평가·HF 업로드를 시작하지 않는다.", "",
    ]
    return "\n".join(lines) + "\n"
