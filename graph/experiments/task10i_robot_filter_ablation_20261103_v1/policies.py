"""Explicit experimental filter ablations; no shared Graph source is modified."""
import numpy as np


def confirmed_only(masks,result):
    masks=np.asarray(masks,bool).copy()
    rejected=(masks[0]&masks[1])|result['confirmed_robot']
    masks[:,rejected]=False
    return masks,rejected


def no_robot_filter(camera,surfaces,**_):
    shape=camera.depth_raw_mm.shape
    return dict(confirmed_robot=np.zeros(shape,bool),uncertain=np.zeros(shape,bool))


def apply(ctx,mode):
    import graph.plan_a_geometry.core as core
    if mode=='full':return
    if mode=='confirmed_only':ctx.patch(core,'filtered_masks',confirmed_only);return
    if mode=='off':ctx.patch(core,'classify_robot',no_robot_filter);return
    raise ValueError('unknown filter ablation')
