"""Load the pinned experts and smoke-test routing with synthetic detector outputs."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from graph.five_mode_hard_routing import build_hard_routing_pipeline
from utils.experiment_logger import ExperimentLogger


def synthetic_detection(peg_points, rim_count):
    """Supply a detector-shaped observation without opening any dataset or GT."""
    peg = np.asarray(peg_points, dtype=np.float32).reshape(-1, 3)
    # The rim geometry is synthetic; these coordinates are not evaluation targets.
    rim = np.asarray(
        [[0.08, index * 0.01, 0] for index in range(rim_count)], dtype=np.float32,
    ).reshape(-1, 3)
    return SimpleNamespace(
        xyz=torch.from_numpy(np.concatenate((peg, rim))),
        object_id=torch.tensor([0] * len(peg) + [1] * rim_count),
        metadata=[{"contour_hole": False} for _ in peg]
        + [{"contour_hole": True} for _ in rim],
    )


def evaluate(config):
    """Exercise both real checkpoints and C3 abstention, without measuring accuracy."""
    with ExperimentLogger(
        team="graph", model="vertex_edge_gated_mpnn", config=config, job_type="evaluate",
    ) as logger:
        pipeline = build_hard_routing_pipeline(config)
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        logger.record(
            device=str(pipeline.device), torch_version=str(torch.__version__),
            evaluator_sha256=file_sha256(Path(__file__)),
            pipeline_sha256=file_sha256(Path(__file__).parents[2] / "graph/five_mode_hard_routing.py"),
            detector_executed=False, synthetic_observations=True,
        )
        for settings in config["checkpoints"].values():
            logger.link_output(settings["path"], checkpoint=True)

        cases = {
            "condition_1": synthetic_detection(
                [[0, 0, 0], [0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.02]], 3,
            ),
            "condition_2": synthetic_detection([[0, 0, 0], [0.03, 0, 0]], 1),
            "condition_3": synthetic_detection([[0, 0, 0]], 1),
        }
        surface = np.asarray([[0.01, 0.01, 0.01]], dtype=np.float32)
        tcp = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
        results = {}
        for condition, detection in cases.items():
            prediction = pipeline.predict_detection(detection, surface, tcp)
            if prediction.condition != condition:
                raise AssertionError(f"Expected {condition}, got {prediction.condition}")
            if condition == "condition_3":
                if prediction.object21 is not None or prediction.selected_model is not None:
                    raise AssertionError("C3 must return unavailable without running an expert")
            elif prediction.selected_model != condition or prediction.object21.shape != (21,):
                raise AssertionError(f"Expected one finite Object21 from the {condition} expert")
            summary = asdict(prediction)
            summary["object21"] = (
                None if prediction.object21 is None else prediction.object21.cpu().tolist()
            )
            results[condition] = summary
            logger.log({"passed": 1}, step=len(results), prefix=condition)

        result_path = output / "results.json"
        write_json(result_path, dict(
            run_id=logger.run_id, config=config, device=str(pipeline.device),
            synthetic_observations=True, detector_executed=False, results=results,
        ))
        logger.link_output(result_path, kind="metrics")
        print(f"Passed C1/C2 checkpoint inference and C3 abstention: {result_path}", flush=True)
        return output
