"""One current GPU robot classification shared by CPU correction and final masks."""
import torch


class RobotFilterFrame:
    def __init__(self, system, cameras, diagnostics):
        self.system = system
        self.identity = tuple((c.episode, c.timestep, c.camera, c.timestamp) for c in cameras)
        self.diagnostics = diagnostics
        self._host = None
        self._closed = False

    def _check(self):
        if self._closed:
            raise RuntimeError('robot classification belongs to a completed frame')

    def clean(self, masks):
        self._check()
        if set(masks) != set(self.diagnostics):
            raise ValueError('robot classification camera identity')
        result = {}
        for name, raw in masks.items():
            robot = self.diagnostics[name]
            confirmed, uncertain = robot['confirmed_robot'], robot['uncertain']
            if (not isinstance(raw, torch.Tensor) or raw.device != confirmed.device
                    or raw.dtype != torch.bool or tuple(raw.shape) != (2, *confirmed.shape)):
                raise ValueError('current CUDA bool masks required')
            rejected = confirmed | uncertain | (raw[0] & raw[1])
            result[name] = raw & ~rejected
        return result

    def prepare(self, cameras, data, masks, surfaces):
        """Explicit host correction boundary; never recompute robot geometry."""
        self._check()
        identity = tuple((c.episode, c.timestep, c.camera, c.timestamp) for c in cameras)
        if identity != self.identity or str(data['episode']) != identity[0][0] or int(data['timestep']) != identity[0][1]:
            raise ValueError('foreign robot classification frame')
        if self._host is None:
            keys = ('confirmed_robot', 'uncertain', 'candidate', 'predicted_robot_depth',
                    'link_owner', 'inside_known_link_volume')
            self._host = {}
            for name, result in self.diagnostics.items():
                value = {key: result[key].detach().cpu().numpy() for key in keys}
                for array in value.values():
                    array.setflags(write=False)
                value.update(method='per-shape collision CV-depth agreement or measured point inside actual link collision volume v2',
                             backend='cuda',
                             tolerance_m=.0025, uncertainty_band_m=.006)
                self._host[name] = value
        # The unchanged gate may wrap its host masks with torch.as_tensor.
        # Give it writable owned arrays when the raw correction input is read-only.
        writable={name:(value if value.flags.writeable else value.copy()) for name,value in masks.items()}
        return self.system.prepare(cameras, data, writable, surfaces, robot_diagnostics=self._host)

    def close(self):
        self._closed = True
        self._host = None
