"""Use the pinned SAM2 video predictor with one resident input frame."""

from collections import OrderedDict

import numpy as np
import torch


def configure_memory_slots(predictor, slots):
    """Keep the trained embeddings for the prompt and most recent frames."""
    if type(slots) is not int or slots not in (1, 3, 7):
        raise ValueError("SAM2 spatial memory slots must be 1, 3, or 7")
    embeddings = getattr(predictor, "maskmem_tpos_enc", None)
    if (getattr(predictor, "num_maskmem", None) != 7
            or not isinstance(embeddings, torch.nn.Parameter)
            or embeddings.ndim != 4 or embeddings.shape[0] != 7):
        raise ValueError("expected an unmodified seven-slot SAM2 checkpoint")
    if slots == 7:
        return
    # The newest embeddings are at the start. The prompt embedding is last.
    selected = torch.cat((embeddings.detach()[:slots - 1], embeddings.detach()[-1:]), dim=0)
    predictor.maskmem_tpos_enc = torch.nn.Parameter(
        selected, requires_grad=embeddings.requires_grad
    )
    predictor.num_maskmem = slots


def build_predictor(checkpoint, image_size, memory_slots):
    """Load SAM2.1 Small and select spatial memory slots after loading weights."""
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(checkpoint),
        device="cuda",
        hydra_overrides_extra=[f"++model.image_size={image_size}"],
    )
    configure_memory_slots(predictor, memory_slots)
    # Return the raw masks, matching the benchmark and the extension-free install.
    predictor.fill_hole_area = 0
    return predictor


def initialize_state(predictor, pixels, image_shape, boxes):
    """Mirror the pinned predictor's init_state after its video loader."""
    device = torch.device(predictor.device)
    height, width = image_shape
    state = {
        "images": [pixels],
        "num_frames": 1,
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
        "video_height": height,
        "video_width": width,
        "device": device,
        "storage_device": device,
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
    }
    predictor._get_image_feature(state, frame_idx=0, batch_size=1)
    try:
        for object_id, box in boxes.items():
            coords = np.asarray(box, dtype=np.float32)
            if coords.shape != (4,) or not np.isfinite(coords).all():
                raise ValueError(f"expected four finite XYXY coordinates for object {object_id}")
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=int(object_id), box=coords)
    except BaseException:
        predictor.reset_state(state)
        raise
    return state
