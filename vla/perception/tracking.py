"""SAM2 video tracking for runtime visual pipelines.

Example::

    tracker = SAM2Tracker()
    state = tracker.start(video_path, {1: [x1, y1, x2, y2]})
    for frame_idx, masks in tracker.track(state):
        ...
    tracker.reset(state)

``video_path`` is a directory of RGB frames accepted by SAM2. Boxes use pixel
coordinates in ``[x1, y1, x2, y2]`` format and integer object IDs. Returned
masks are keyed by tracker ID; each value is a boolean Torch tensor with shape
``[height, width]``. ``vla.perception.rgbd.attach_rgbd`` joins this iterator to
recorded RGB-D, camera calibration, and TCP poses. The Visual output includes
explicit conversion from tracker IDs to the object IDs consumed by Graph.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2] / "visual"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_small.pt"


class SAM2Tracker:
    """Load a SAM2 video predictor once and track prompted objects in videos.

    Args:
        device: Torch device for SAM2 inference. Defaults to ``"cuda"`` to
            match the existing Visual pipeline.
        config: SAM2 Hydra config path, relative to the SAM2 package config
            directory or an absolute path.
        checkpoint: Model checkpoint path.
        hydra_overrides: Optional SAM2 model configuration overrides.
        predictor: Existing SAM2-compatible predictor, useful when another
            pipeline owns model construction. When provided, model arguments
            are ignored.
    """

    def __init__(self, device: str = "cuda", config: str = DEFAULT_CONFIG,
                 checkpoint: str | Path = DEFAULT_CHECKPOINT,
                 hydra_overrides: Sequence[str] | None = None, predictor=None):
        self.device = device
        if predictor is None:
            from sam2.build_sam import build_sam2_video_predictor

            predictor = build_sam2_video_predictor(
                config,
                str(checkpoint),
                device=device,
                hydra_overrides_extra=list(hydra_overrides or ()),
            )
        self.predictor = predictor

    def start(self, video_path: str | Path,
              boxes: Mapping[int, Sequence[float]], frame_idx: int = 0) -> dict:
        """Create a video state and prompt each object with its initial box."""
        state = self.predictor.init_state(video_path=str(video_path))
        for object_id, box in boxes.items():
            coords = np.asarray(box, dtype=np.float32)
            if coords.shape != (4,):
                self.predictor.reset_state(state)
                raise ValueError(f"box for object {object_id} must have four coordinates")
            self.predictor.add_new_points_or_box(
                state, frame_idx=frame_idx, obj_id=int(object_id), box=coords
            )
        return state

    def start_frames(self,images,image_shape,boxes,frame_idx=0):
        """Initialize the installed SAM2 video-state contract from resident frames.

        Mirrors SAM2VideoPredictor.init_state after its image-loader boundary.
        No directory loader, JPEG encoding, predictor patch or future RGB input.
        """
        import torch
        from collections import OrderedDict
        if not 0<=frame_idx<len(images):raise ValueError('observed prompt frame required')
        device=torch.device(self.predictor.device);sample=images[frame_idx]
        if sample.device!=torch.empty(0,device=device).device or sample.shape!=(3,self.predictor.image_size,self.predictor.image_size):
            raise ValueError('SAM2 resident normalized image/device mismatch')
        state=dict(images=images,num_frames=len(images),offload_video_to_cpu=False,offload_state_to_cpu=False,
            video_height=int(image_shape[0]),video_width=int(image_shape[1]),device=device,storage_device=device,
            point_inputs_per_obj={},mask_inputs_per_obj={},cached_features={},constants={},
            obj_id_to_idx=OrderedDict(),obj_idx_to_id=OrderedDict(),obj_ids=[],
            output_dict_per_obj={},temp_output_dict_per_obj={},frames_tracked_per_obj={})
        self.predictor._get_image_feature(state,frame_idx=0,batch_size=1)
        try:
            for object_id,box in boxes.items():
                coords=np.asarray(box,dtype=np.float32)
                if coords.shape!=(4,) or not np.isfinite(coords).all():raise ValueError('finite XYXY box required')
                self.predictor.add_new_points_or_box(state,frame_idx=frame_idx,obj_id=int(object_id),box=coords)
        except BaseException:self.predictor.reset_state(state);raise
        return state

    def track(self, state: dict, start_frame: int = 0, *, mask_filter=None) -> Iterator[tuple[int, dict[int, object]]]:
        """Yield masks on the predictor device.

        Optional ``mask_filter(frame_index, masks)`` runs before any CPU adapter.
        It must return boolean masks with identical IDs, shapes and devices.
        When it exposes prepare_frame/cancel_prepared and state has num_frames,
        prepare_frame runs before advancing each forward SAM2 frame. This
        permits CPU robot geometry to overlap inference; output indices must
        remain consecutive. The predictor itself stays on this thread.
        Without it the existing tracker behavior is unchanged.
        """
        prepare=getattr(mask_filter,'prepare_frame',None);cancel=getattr(mask_filter,'cancel_prepared',None)
        count=state.get('num_frames');expected=int(start_frame)
        has_preparation=callable(prepare) and callable(cancel)
        ahead=has_preparation and isinstance(count,int)
        frames=iter(self.predictor.propagate_in_video(state,start_frame_idx=start_frame))
        try:
            while True:
                if ahead and expected<count:prepare(expected)
                try:frame_idx,object_ids,logits=next(frames)
                except StopIteration:break
                if ahead and int(frame_idx)!=expected:raise ValueError('prepared SAM2 tracking requires consecutive forward frames')
                masks={int(object_id):logits[i,0]>0 for i,object_id in enumerate(object_ids)}
                if mask_filter is not None:
                    import torch
                    filtered=mask_filter(int(frame_idx),masks)
                    if set(filtered)!=set(masks):raise ValueError('mask_filter changed tracker object IDs')
                    for object_id,mask in filtered.items():
                        original=masks[object_id]
                        if (not isinstance(mask,torch.Tensor) or mask.dtype!=torch.bool
                                or mask.shape!=original.shape or mask.device!=original.device):
                            raise ValueError('mask_filter must preserve mask shape, dtype and device')
                    masks=filtered
                expected=int(frame_idx)+1
                yield int(frame_idx),masks
        finally:
            if has_preparation:cancel()
            close=getattr(frames,'close',None)
            if close is not None:close()

    def reset(self, state: dict) -> None:
        """Release the predictor's per-video tracking state."""
        self.predictor.reset_state(state)


# Keep the concise spelling available alongside the descriptive class name.
Tracker = SAM2Tracker
