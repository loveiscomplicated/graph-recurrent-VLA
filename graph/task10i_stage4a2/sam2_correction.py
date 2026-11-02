"""Transactional correction for the audited installed SAM2 predictor contract.

Already-tracked frames default to non-conditioning outputs. Re-propagating that
same frame discards its new prompt in the installed version. Keep the immediate
prompt result and preflight once; propagate the next actual observation later.
"""
import hashlib
import numpy as np
import torch


def fork_containers(value):
    """Fork mutable state containers, sharing immutable tensors/images.

    This is not a general-purpose serialization of SAM2 memory. Isolation is
    verified against full tensor bytes in the runtime control before adoption.
    """
    if isinstance(value,dict):return type(value)((k,fork_containers(v)) for k,v in value.items())
    if isinstance(value,list):return [fork_containers(v) for v in value]
    if isinstance(value,tuple):return tuple(fork_containers(v) for v in value)
    return value


def state_digest(state):
    digest=hashlib.sha256()
    def visit(value,path):
        digest.update(path.encode())
        if torch.is_tensor(value):
            x=value.detach().contiguous().cpu()
            digest.update(str((x.shape,x.dtype)).encode());digest.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value,dict):
            for k,v in value.items():visit(v,path+'/'+str(k))
        elif isinstance(value,(list,tuple)):
            for i,v in enumerate(value):visit(v,path+'/'+str(i))
        elif hasattr(value,'frames'):
            for i,v in enumerate(value.frames):visit(v,path+'/observed_image/'+str(i))
        else:digest.update(repr(value).encode())
    visit(state,'state')
    return digest.hexdigest()


@torch.inference_mode()
def propose(adapter,object_id,timestep,box,conditioning=False,verify_isolation=False):
    """Return an uncommitted state/mask transaction for current predicted box.

    No GT, actor ID, expert phase, future image or source episode is accepted.
    `object_id` is the semantic tracker slot1/2, not simulator Actor identity.
    """
    if timestep!=len(adapter.rgb)-1:raise ValueError('correction requires the current observed frame')
    coords=np.asarray(box,np.float32)
    h,w=adapter.rgb[-1].shape[:2]
    if coords.shape!=(4,) or not np.isfinite(coords).all() or not (
        0<=coords[0]<coords[2]<=w and 0<=coords[1]<coords[3]<=h):
        raise ValueError('finite in-image pixel xyxy prompt required')
    group=next((g for g in adapter.states if object_id in g),None)
    if group is None:return None,dict(status='NO_INITIALIZED_OBJECT_SLOT')
    original=adapter.states[group]
    before=state_digest(original) if verify_isolation else None
    candidate=fork_containers(original)
    predictor=adapter.tracker.predictor
    old_flag=predictor.add_all_frames_to_correct_as_cond
    try:
        predictor.add_all_frames_to_correct_as_cond=conditioning
        frame,ids,logits=predictor.add_new_points_or_box(candidate,frame_idx=timestep,
            obj_id=object_id,box=coords,clear_old_points=True)
    finally:predictor.add_all_frames_to_correct_as_cond=old_flag
    assert frame==timestep and list(ids)==list(group)
    masks={int(oid):(logits[i,0]>0).cpu().numpy() for i,oid in enumerate(ids)}
    predictor.propagate_in_video_preflight(candidate)
    if verify_isolation:
        assert state_digest(original)==before,'SAM2 candidate mutated the original state'
    index=candidate['obj_id_to_idx'][object_id]
    storage='cond_frame_outputs' if timestep in candidate['output_dict_per_obj'][index]['cond_frame_outputs'] else 'non_cond_frame_outputs'
    return dict(group=group,state=candidate,masks=masks,original=original,
        prompt_logits=logits.detach().float().cpu().numpy()),dict(
        status='UNCOMMITTED_CANDIDATE',prompt_frame=timestep,conditioning=conditioning,
        same_frame_repropagated=False,storage_key=storage,original_state_sha256=before,
        original_tensor_bytes_unchanged=bool(verify_isolation),global_predictor_flag_restored=True)


def commit(adapter,transaction,accepted):
    if transaction is None:raise ValueError('missing candidate transaction')
    group=transaction['group']
    if adapter.states.get(group) is not transaction['original']:
        raise ValueError('live memory changed since candidate proposal')
    if accepted:adapter.states[group]=transaction['state']
    # Rejection keeps the exact original memory object. No in-place rollback
    # approximation or replay of only the suffix is performed.
    return bool(accepted)
