"""Reuse the paired evaluator with an observation-only corner-ID input path."""

from data_collection.episode_dataset import file_sha256
import evaluate_mamba
from gtfree_frontend import GTFreeFrontend
import gtfree_frontend
import match
import box_matcher

ORIGINAL_LOGGER = evaluate_mamba.ExperimentLogger


class CameraOnlyFrontend(GTFreeFrontend):
    def step(self, obs, tcp, gt_match_only=None):
        camera=obs['sensor_data']['base_camera']
        rgbd={'sensor_data':{'base_camera':{key:camera[key] for key in ('rgb','depth')}},
              'sensor_param':obs['sensor_param']}
        return super().step(rgbd,tcp)


def new_logger(*, config, **kwargs):
    config.update(physical_ids='detector geometry + TCP; no object GT',
                  wrapper_sha256=file_sha256(__file__),
                  id_code_sha256={module.__name__:file_sha256(module.__file__)
                                      for module in (gtfree_frontend,match,box_matcher)})
    kwargs['model']='mamba_gtfree_corner_id_eval'
    return ORIGINAL_LOGGER(config=config,**kwargs)


if __name__=='__main__':
    evaluate_mamba.fixed.HybridFrontend=CameraOnlyFrontend
    evaluate_mamba.ExperimentLogger=new_logger
    evaluate_mamba.main()
