"""Encode the unmodified main camera for each rollout, with an on-video status overlay."""
from pathlib import Path
import subprocess

import cv2
import numpy as np


class RolloutVideo:
    def __init__(self, folder, seed, mode, fps):
        self.folder, self.seed, self.mode, self.fps = folder, seed, mode, fps
        self.partial = folder / f'{seed}_{mode}_partial.mp4'
        self.nframes = 0
        self.encoder = subprocess.Popen(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
             '-pixel_format', 'rgb24', '-video_size', '512x512', '-framerate', str(fps),
             '-i', '-', '-an', '-c:v', 'libx264', '-threads', '1', '-preset', 'veryfast',
             '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.partial)],
            stdin=subprocess.PIPE,
        )

    def add(self, obs, step, status, moved_mm):
        frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()[:, :, :3]
        frame = np.ascontiguousarray(frame, dtype=np.uint8).copy()
        assert frame.shape == (512, 512, 3)
        cv2.rectangle(frame, (0, 0), (511, 51), (22, 22, 22), -1)
        title = 'New frontend' if self.mode == 'hybrid' else 'Existing frontend'
        cv2.putText(frame, f'{title} | seed {self.seed} | step {step}',
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .52, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(frame, f'{status} | Box offset {moved_mm:.1f} mm',
                    (10, 43), cv2.FONT_HERSHEY_SIMPLEX, .52, (230, 230, 230), 1, cv2.LINE_AA)
        self.encoder.stdin.write(frame.tobytes())
        self.nframes += 1
        return frame

    def finish(self, obs, steps, success, reason, moved_mm):
        label = 'SUCCESS' if success else f'FAILED - {reason.upper()}'
        final = self.add(obs, steps, label, moved_mm).tobytes()
        for _ in range(max(1, round(self.fps)) // 2):
            self.encoder.stdin.write(final)
            self.nframes += 1
        self.encoder.stdin.close()
        if self.encoder.wait() != 0 or not self.partial.is_file():
            raise RuntimeError(f'MP4 encoding failed for {self.seed} / {self.mode}')
        verdict = 'success' if success else 'fail'
        path = self.folder / f'{self.seed}_{self.mode}_{verdict}.mp4'
        self.partial.replace(path)
        return path, self.nframes

    def close(self):
        if self.encoder.poll() is None:
            self.encoder.stdin.close()
            self.encoder.wait()
