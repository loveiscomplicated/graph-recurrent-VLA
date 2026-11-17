"""pose_v1 빠른 경로 자체 검사: 직전 자세가 앞뒤로 뒤집혀 있으면 빠른 경로가 거부하고 전체 탐색으로 바로잡아야 한다.
Usage: python scripts/test_pose_v1.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_v1 as pv  # noqa: E402


def peg_points(L=0.1, r=0.02, n=600, seed=0):
    """윗면(+z)만 보이는 peg 표면 점 (원점, 축 = world x, head = x > 0). 끝면이 안 보이면 표면 잔차만으로는
    앞뒤 뒤집힌 자세와 구분이 안 되고 색 잔차만 다르다 (빠른 경로 판정이 색 항을 봐야 하는 경우)."""
    rng = np.random.default_rng(seed)
    P = np.c_[rng.uniform(-L, L, n), rng.uniform(-r, r, n), np.full(n, r)] + [0, 0, 0.05]  # 테이블(z=0) 위
    return P, P[:, 0] > 0


def test_flipped_prev_is_corrected():
    P, head = peg_points()
    pe = pv.PegEstimator(0.1, 0.02, fast=True)
    flipped = np.eye(4)
    flipped[:3, :3] = np.diag([-1.0, -1.0, 1.0])  # 축 180° 뒤집힘
    flipped[:3, 3] = [0, 0, 0.05]
    pe.prev, pe.prev_mean = flipped, P.mean(0)
    T, _ = pe(P, head)
    assert T is not None and T[0, 0] > 0.99, f"head 방향이 +x 여야 함, got axis {T[:3, 0]}"
    assert np.linalg.norm(T[:3, 3] - [0, 0, 0.05]) < 0.003, f"중심 오차 {T[:3, 3]}"


def test_good_prev_uses_fast_path():
    P, head = peg_points()
    pe = pv.PegEstimator(0.1, 0.02, fast=True)
    good = np.eye(4)
    good[:3, 3] = [0.001, 0, 0.05]
    pe.prev, pe.prev_mean = good, P.mean(0)
    T, _ = pe(P, head)
    assert T[0, 0] > 0.99 and np.linalg.norm(T[:3, 3] - [0, 0, 0.05]) < 0.002


if __name__ == "__main__":
    test_flipped_prev_is_corrected()
    test_good_prev_uses_fast_path()
    print("ok")
