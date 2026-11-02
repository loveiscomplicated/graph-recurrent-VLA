"""Experimental causal correction over unchanged current-frame baseline masks.

No GT, expert phase, actor pose, future RGB-D or failure-time input exists here.
The old found-state behavior is preserved in B0 and never retroactively changed.
Fresh SAM2 memories are validated before swapping a semantic object's override.
"""
from dataclasses import asdict,replace
import hashlib
import numpy as np
from .reanchor_state_machine import Evidence,Rules,ReliabilitySlot
from graph.persistent_geometry.contracts import synchronized
from visual.runtime.initialization import box_iou


def bbox(mask):
    y,x=np.nonzero(mask)
    return [float(x.min()),float(y.min()),float(x.max()+1),float(y.max()+1)] if len(x) else None


class CausalCorrection:
    def __init__(self,detector,tracker,geometry,features=None,wrist_prompt='red and white block',
                 rules=None,anchor_factory=None,uncertain_output_policy='suppress',lazy_matching=False,correction_slots=None,
                 late_baseline_handoff=False):
        if uncertain_output_policy not in ('suppress','preserve_current'):
            raise ValueError('unknown uncertainty output policy')
        self.uncertain_output_policy=uncertain_output_policy
        self.lazy_matching=bool(lazy_matching)
        available={('base_camera',0),('hand_camera',0),('base_camera',1)}
        self.correction_slots=available if correction_slots is None else set(map(tuple,correction_slots))
        if not self.correction_slots<=available:raise ValueError('unsupported correction scope')
        self.late_baseline_handoff=bool(late_baseline_handoff)
        if anchor_factory is None:
            from graph.task10i_stage4a.reliable_reanchor import AnchorState
            anchor_factory=AnchorState
        self.detector=detector;self.tracker=tracker;self.geometry=geometry;self.features=features
        self.wrist_prompt=wrist_prompt;self.anchor_factory=anchor_factory
        # This is a multi-evidence Development hypothesis, not a calibrated
        # detector probability or a threshold selected using Internal/Final.
        self.rules=rules or Rules(minimum_detection_score=.2)
        self.active={};self.identity=None

    def reset(self,episode):
        self.close();self.identity=str(episode);self.last_t=-1
        self.slots={(c,o):ReliabilitySlot(episode,c,o,self.rules) for c,o in
                    [('base_camera',0),('hand_camera',0),('base_camera',1)]}
        self.pending={};self.centers={};self.references={};self.events=[]
        self.candidate_masks=[];self.reference_provenance=[]
        self.baseline_found={};self.early_anchors={};self.handoffs=[]
        if self.features is not None:self.features.reset(episode)

    def close(self):
        for state in self.active.values():state.close()
        self.active={}

    @staticmethod
    def presence(state):
        if state is None or not hasattr(state,'state'):return None
        local=state.last_t-state.anchor_t;memory=state.state
        index=memory['obj_id_to_idx'][1];outputs=memory['output_dict_per_obj'][index]
        out=outputs['cond_frame_outputs'].get(local,outputs['non_cond_frame_outputs'].get(local))
        return float(out['object_score_logits'].detach().float().cpu().item()) if out is not None else None

    def evidence(self,camera,oid,masks,prepared,t,key,presence=None):
        mask=masks[camera.camera][oid];valid=mask&np.isfinite(camera.depth)
        robot=prepared['filters'][camera.camera]
        nonrobot=valid&~robot['confirmed_robot']&~robot['uncertain']
        clean=prepared['clean'][camera.camera][oid]&np.isfinite(camera.depth)
        center=np.median(camera.world_points()[clean],axis=0) if clean.any() else None
        previous=self.centers.get(key);consistent=True
        if center is not None and previous is not None:
            prior_t,prior_center=previous
            consistent=bool(np.linalg.norm(center-prior_center)<=.08+.01*max(0,t-prior_t-1))
        e=Evidence(int(mask.sum()),float(valid.sum()/mask.sum()) if mask.any() else 0.,
            float(nonrobot.sum()/valid.sum()) if valid.any() else 0.,int(clean.sum()),
            object_score_logit=presence,temporal_consistent=consistent)
        return e,clean,center

    def feature(self,camera,t,mask):
        return self.features.embed(camera,t,mask) if self.features is not None else None

    def maybe_reference(self,key,t,clean,evidence,center):
        slot=self.slots[key]
        if self.features is None or key in self.references or slot.good_run<3 or not slot.ever_found:
            return
        if evidence.nonrobot_fraction<.9 or evidence.supported_pixels<6:return
        vector=self.feature(key[0],t,clean)
        if vector is None:return
        self.references[key]=vector.copy()
        self.reference_provenance.append(dict(episode=self.identity,camera=key[0],object_id=key[1],
            timestep=t,mask_bits=np.packbits(clean.ravel()).tolist(),embedding=vector.tolist(),
            norm=float(np.linalg.norm(vector)),approval='three consecutive current support passes; predicted nonrobot fraction>=.9',
            policy='first supported reference frozen; no self-updating bank',GT_used=False,
            origin='accepted_fresh_memory' if key in self.active else 'existing_baseline_prediction'))

    def process(self,cameras,data,surfaces,baseline_masks,ever_found):
        synchronized(cameras);t=int(data['timestep']);episode=str(data['episode'])
        if {c.camera for c in cameras}!={'base_camera','hand_camera'}:raise ValueError('two named cameras required')
        if episode!=self.identity or t!=self.last_t+1:raise ValueError('explicit causal episode reset required')
        if any(c.episode!=episode or c.timestep!=t for c in cameras):raise ValueError('camera clock differs')
        if set(baseline_masks)!={'base_camera','hand_camera'}:raise ValueError('two current cameras required')
        original={c:m.copy() for c,m in baseline_masks.items()}
        for camera in cameras:
            if original[camera.camera].dtype!=bool or original[camera.camera].shape!=(2,*camera.rgb.shape[:2]):
                raise ValueError('current bool semantic masks required')
            if self.features is not None:self.features.observe(episode,camera.camera,t,camera.rgb)
        self.last_t=t;current={c:m.copy() for c,m in original.items()};lookup={c.camera:c for c in cameras}
        # Advance each accepted replacement exactly once, with current RGB only.
        for key,state in self.active.items():
            self.tracker.research_context=dict(episode=episode,camera=key[0],object_id=key[1],timestep=t,
                anchor_t=state.anchor_t,kind='propagate_override')
            state.append(lookup[key[0]].rgb,t);current[key[0]][key[1]]=state.current()
        prepared=self.geometry.system.prepare(cameras,data,current,surfaces)
        if self.late_baseline_handoff:
            for key in list(self.active):
                newly_initialized=bool(ever_found.get(key,False)) and not self.baseline_found.get(key,False)
                if not newly_initialized or not self.early_anchors.get(key,False):continue
                c,oid=key;camera=lookup[c];slot=self.slots[key]
                trial={name:m.copy() for name,m in current.items()};trial[c][oid]=original[c][oid]
                proposed=self.geometry.system.prepare(cameras,data,trial,surfaces)
                evidence,clean,center=self.evidence(camera,oid,trial,proposed,t,key)
                old_clean=prepared['clean'][c][oid]&np.isfinite(camera.depth)
                overlap=float((old_clean&clean).sum()/old_clean.sum()) if old_clean.any() else 0.
                grows=bool(clean.sum()>=old_clean.sum() and old_clean.any())
                supported=slot.usable(evidence) and evidence.nonrobot_fraction>=.9 and overlap>=.8 and grows
                reference=self.references.get(key);similarity=None;negative_similarity=None
                if supported and reference is not None and self.features is not None:
                    vector=self.feature(c,t,clean);negative=[]
                    if vector is not None:
                        similarity=float(vector@reference)
                        opposite=self.references.get((c,1-oid))
                        robot=self.feature(c,t,proposed['filters'][c]['confirmed_robot'])
                        for value in (opposite,robot):
                            if value is not None:negative.append(float(vector@value))
                    negative_similarity=max(negative) if negative else None
                reid_pass=(reference is None or (similarity is not None and similarity>=self.rules.minimum_reid_similarity and
                    (negative_similarity is None or similarity-negative_similarity>=self.rules.minimum_reid_margin)))
                accepted=bool(supported and reid_pass)
                self.handoffs.append(dict(episode=episode,timestep=t,camera=c,object_id=oid,
                    source='new original DINO-confirmed initialization after an earlier experimental anchor',
                    evidence=asdict(evidence),old_clean_pixels=int(old_clean.sum()),new_clean_pixels=int(clean.sum()),
                    old_support_overlap=overlap,reference_available=reference is not None,similarity=similarity,
                    negative_similarity=negative_similarity,reid_pass=reid_pass,accepted=accepted,
                    future_or_GT_trigger=False,mode='resume_already_running_original_SAM_memory'))
                if accepted:
                    self.active.pop(key).close();self.early_anchors.pop(key,None)
                    current=trial;prepared=proposed
        self.baseline_found={key:bool(ever_found.get(key,False)) for key in self.slots}
        for key,slot in self.slots.items():
            c,oid=key;camera=lookup[c]
            evidence,clean,center=self.evidence(camera,oid,current,prepared,t,key,self.presence(self.active.get(key)))
            observed_due=slot.observe(episode,c,oid,t,evidence,initialized=bool(ever_found.get(key,False)))
            scope_enabled=key in self.correction_slots
            due=observed_due and scope_enabled
            event=dict(episode=episode,timestep=t,camera=c,object_id=oid,evidence=asdict(evidence),
                ever_found=slot.ever_found,currently_valid=slot.currently_valid,detection_requested=due,
                accepted=False,proposals=[],reference_available=key in self.references,
                scope_enabled=scope_enabled,redetection_due_but_scope_disabled=observed_due and not scope_enabled)
            if slot.currently_valid:
                if center is not None:self.centers[key]=(t,center)
                self.pending.pop(key,None)
                self.maybe_reference(key,t,clean,evidence,center)
            if due:
                prompt='cardboard box' if oid==1 else self.wrist_prompt if c=='hand_camera' else 'red and white block'
                target=getattr(self.detector,'detector',self.detector)
                if hasattr(target,'context'):target.context.update(episode_id=episode,camera=c,decision_timestep=t,object_id=oid)
                boxes,scores=self.detector(camera.rgb,prompt)
                rgb_hash=hashlib.sha256(camera.rgb.tobytes()).hexdigest()
                source_event=next((r for r in reversed(getattr(target,'events',[]))
                    if r['input_rgb_sha256']==rgb_hash and r['prompt']==prompt),None)
                event['prompt']=prompt;event['input_rgb_sha256']=rgb_hash
                event['raw_query_artifact']=source_event['raw_path'] if source_event else None
                other=bbox(current[c][1-oid]);prior=self.pending.get(key)
                candidates=[];all_states=[];fresh_boxes=[]
                robot_embedding=None
                # A missing past reference bypasses Re-ID in the original
                # decision rule. Keep reference creation unchanged, but allow
                # an explicit operational switch to skip unused comparison
                # features. Frozen earlier conditions retain eager behavior.
                compare_features=not self.lazy_matching or key in self.references
                if self.features is not None and compare_features and not self.lazy_matching:
                    robot_embedding=self.feature(c,t,prepared['filters'][c]['confirmed_robot'])
                robot_feature_requested=False
                try:
                    for rank,(raw,score) in enumerate(zip(boxes,scores)):
                        box=np.clip(np.asarray(raw,float),[0,0,0,0],
                                    [camera.rgb.shape[1],camera.rgb.shape[0]]*2).tolist()
                        area=(box[2]-box[0])*(box[3]-box[1])
                        if area<=0 or area>=.25*camera.rgb.shape[0]*camera.rgb.shape[1]:continue
                        if len(all_states)>=5:break
                        if score<self.rules.minimum_detection_score:continue
                        if other is not None and box_iou(box,other)>=.3:continue
                        self.tracker.research_context=dict(episode=episode,camera=c,object_id=oid,timestep=t,
                            anchor_t=t,kind='candidate_prompt',post_rank=rank)
                        state=self.anchor_factory(self.tracker,camera.rgb,box,t);all_states.append(state)
                        mask=state.current()
                        trial={name:value.copy() for name,value in current.items()};trial[c][oid]=mask
                        trial_prepared=self.geometry.system.prepare(cameras,data,trial,surfaces)
                        candidate_evidence,candidate_clean,candidate_center=self.evidence(camera,oid,trial,trial_prepared,t,key,self.presence(state))
                        supported=slot.usable(candidate_evidence)
                        temporal=bool(prior is not None and 0<t-prior[0]<=10 and any(box_iou(box,b)>=.5 for b in prior[1]))
                        vector=self.feature(c,t,candidate_clean) if supported and compare_features else None
                        reference=self.references.get(key)
                        similarity=float(vector@reference) if vector is not None and reference is not None else None
                        if self.lazy_matching and vector is not None and reference is not None and not robot_feature_requested:
                            robot_embedding=self.feature(c,t,prepared['filters'][c]['confirmed_robot'])
                            robot_feature_requested=True
                        opposite=self.references.get((c,1-oid));negative=[]
                        if vector is not None:
                            for v in (opposite,robot_embedding):
                                if v is not None:negative.append(float(vector@v))
                        negative_similarity=max(negative) if negative else None
                        reid_pass=(reference is None or (similarity is not None and similarity>=self.rules.minimum_reid_similarity and
                            (negative_similarity is None or similarity-negative_similarity>=self.rules.minimum_reid_margin)))
                        proposal=dict(post_rank=rank,box=box,score=float(score),evidence=asdict(candidate_evidence),
                            temporal_confirmed=temporal,reference_available=reference is not None,
                            similarity=similarity,negative_similarity=negative_similarity,reid_pass=reid_pass,
                            current_support_pass=supported)
                        event['proposals'].append(proposal)
                        self.candidate_masks.append(dict(episode=episode,camera=c,object_id=oid,timestep=t,
                            mask_bits=np.packbits(mask.ravel()).tolist(),**{k:v for k,v in proposal.items() if k!='evidence'}))
                        if supported:fresh_boxes.append(box)
                        if supported and reid_pass:candidates.append((temporal,float(score),state,mask,proposal,candidate_evidence,candidate_center))
                    self.pending[key]=(t,fresh_boxes)
                    if candidates:
                        best=max(candidates,key=lambda x:(x[0],x[1]));temporal,score,state,mask,proposal,ce,cc=best
                        accepted=slot.assess_candidate(t,score,ce,temporal,proposal['reference_available'],
                            proposal['similarity'],proposal['negative_similarity'])
                        if accepted:
                            slot.commit(t,slot.usable(ce))
                            previous=self.active.get(key)
                            self.active[key]=state;all_states.remove(state)
                            self.early_anchors[key]=not bool(ever_found.get(key,False))
                            if previous is not None:previous.close()
                            current[c][oid]=mask;event['accepted']=True;event['chosen']=proposal
                            if cc is not None:self.centers[key]=(t,cc)
                            self.pending.pop(key,None)
                    else:slot.reject(t,'no_current_candidate_passed_prediction_only_support_and_reid')
                finally:
                    for state in all_states:state.close()
            if scope_enabled and not slot.currently_valid and slot.bad_run>=self.rules.uncertain_frames:
                event['uncertain_output_policy']=self.uncertain_output_policy
                if self.uncertain_output_policy=='suppress':
                    current[c][oid]=False;event['invalid_mask_suppressed']=True
                else:
                    # Keep this observation's SAM2 result pending an accepted
                    # replacement. Never reuse a past mask or mark it valid.
                    event['uncertain_current_mask_preserved']=True
            # A preceding semantic correction changes overlap support for the
            # next slot; recompute using only the current predicted masks.
            if event['accepted'] or event.get('invalid_mask_suppressed'):
                prepared=self.geometry.system.prepare(cameras,data,current,surfaces)
            event['result_state']=slot.state.value;self.events.append(event)
        assert np.array_equal(current['hand_camera'][1],original['hand_camera'][1]),'Wrist Box changed'
        assert all(np.array_equal(baseline_masks[c],m) for c,m in original.items()),'baseline masks mutated'
        return current
