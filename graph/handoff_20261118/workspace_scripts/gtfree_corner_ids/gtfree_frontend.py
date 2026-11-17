"""Connect observation-only vertex IDs to the trained full-insertion frontend."""
import numpy as np

from frontend import FullInsertionFrontend
from match import CornerMatcher


def detect_raw(detector, obs):
    base=obs['sensor_data']['base_camera']
    camera=obs['sensor_param']['base_camera']
    return detector(base['rgb'][0].cpu().numpy(),base['depth'][0].cpu().numpy(),
                    camera['intrinsic_cv'][0].cpu().numpy(),camera['extrinsic_cv'][0].cpu().numpy())


def slot_candidates(raw, matcher, tcp):
    points=raw.xyz.cpu().numpy()
    kind=raw.object_id.cpu().numpy()
    rim=np.array([bool(tags.get('contour_hole',False) or tags.get('method')=='box_opening')
                  for tags in raw.metadata])
    family=np.where(kind==0,0,np.where(rim,2,1))
    labels=matcher.step(points,family,tcp)
    mask=labels>=0
    coords=np.zeros((20,3),dtype=np.float32)
    coords[mask]=points[labels[mask]]
    return coords,mask


class GTFreeFrontend(FullInsertionFrontend):
    def reset(self, actor_ids=None):
        super().reset(None)
        self.matcher=CornerMatcher()

    def step(self, obs, tcp, gt_match_only=None):
        self.current_tcp=tcp
        return super().step(obs,tcp,None)

    def detect(self, obs, gt_match_only):
        raw=detect_raw(self.detector,obs)
        coords,mask=slot_candidates(raw,self.matcher,self.current_tcp)
        self.last_observed=mask
        self.last_coordinates=coords
        return coords,mask
