"""Prediction-only per-episode/camera/object reliability state and transactions.

These are experimental Development rules, not calibrated model confidence.
They do not retroactively relabel the original baseline states. A fresh SAM2
candidate must remain separate from live memory until validation and commit.
"""
from dataclasses import dataclass
from enum import Enum
import math


class State(str,Enum):
    UNINITIALIZED='UNINITIALIZED'
    INITIALIZED='INITIALIZED'
    TRACKING_CONFIDENT='TRACKING_CONFIDENT'
    TRACKING_UNCERTAIN='TRACKING_UNCERTAIN'
    LOST='LOST'
    REDETECTION_REQUESTED='REDETECTION_REQUESTED'
    REID_CANDIDATE='REID_CANDIDATE'
    REANCHOR_ACCEPTED='REANCHOR_ACCEPTED'
    REANCHOR_REJECTED='REANCHOR_REJECTED'


@dataclass(frozen=True)
class Evidence:
    mask_pixels:int
    valid_depth_fraction:float
    nonrobot_fraction:float
    supported_pixels:int
    object_score_logit:float|None=None
    temporal_consistent:bool=True

    def valid(self):
        fractions=(self.valid_depth_fraction,self.nonrobot_fraction)
        return (self.mask_pixels>=0 and self.supported_pixels>=0 and
                self.supported_pixels<=self.mask_pixels and
                all(math.isfinite(x) and 0<=x<=1 for x in fractions) and
                (self.object_score_logit is None or math.isfinite(self.object_score_logit)))


@dataclass(frozen=True)
class Rules:
    minimum_pixels:int=6
    minimum_depth_fraction:float=.5
    minimum_nonrobot_fraction:float=.5
    uncertain_frames:int=2
    redetect_interval:int=5
    post_accept_hold:int=10
    minimum_detection_score:float=.3
    minimum_reid_similarity:float=.6
    minimum_reid_margin:float=.05


class ReliabilitySlot:
    def __init__(self,episode,camera,object_id,rules=None):
        if camera not in ('base_camera','hand_camera') or object_id not in (0,1):
            raise ValueError('camera/object contract')
        if camera=='hand_camera' and object_id==1:
            raise ValueError('Wrist Box reanchor is outside the Graph contract')
        self.identity=(str(episode),camera,int(object_id));self.rules=rules or Rules()
        self.state=State.UNINITIALIZED;self.ever_found=False;self.currently_valid=False
        self.last_t=-1;self.bad_run=0;self.good_run=0;self.last_request=-10**9
        self.last_accept=-10**9;self.events=[];self.pending=None

    def _transition(self,state,t,reason):
        previous=self.state;self.state=state
        self.events.append(dict(identity=self.identity,timestep=t,previous=previous.value,
            state=state.value,reason=reason,ever_found=self.ever_found,currently_valid=self.currently_valid))

    def usable(self,evidence):
        r=self.rules
        return (evidence.valid() and evidence.mask_pixels>=r.minimum_pixels and
                evidence.supported_pixels>=r.minimum_pixels and
                evidence.valid_depth_fraction>=r.minimum_depth_fraction and
                evidence.nonrobot_fraction>=r.minimum_nonrobot_fraction and
                evidence.temporal_consistent and
                (evidence.object_score_logit is None or evidence.object_score_logit>0))

    def observe(self,episode,camera,object_id,t,evidence,initialized=False):
        if self.pending is not None:
            raise ValueError('candidate transaction must commit or reject before next observation')
        if (str(episode),camera,int(object_id))!=self.identity or t!=self.last_t+1:
            raise ValueError('episode/camera/object/time discontinuity; replay prefix required')
        if not evidence.valid():raise ValueError('invalid evidence')
        self.last_t=t
        if initialized and not self.ever_found:
            self.ever_found=True;self._transition(State.INITIALIZED,t,'first_actual_initialization')
        self.currently_valid=self.usable(evidence)
        if self.currently_valid:
            self.good_run+=1;self.bad_run=0
            self._transition(State.TRACKING_CONFIDENT,t,'current_prediction_support_pass')
        else:
            self.good_run=0;self.bad_run+=1
            self._transition(State.LOST if self.bad_run>=self.rules.uncertain_frames else
                State.TRACKING_UNCERTAIN,t,'current_prediction_support_fail')
        due=(not self.currently_valid and self.bad_run>=self.rules.uncertain_frames and
             t-self.last_request>=self.rules.redetect_interval and
             t-self.last_accept>=self.rules.post_accept_hold)
        if due:
            self.last_request=t;self._transition(State.REDETECTION_REQUESTED,t,'causal_support_hysteresis')
        return due

    def assess_candidate(self,t,detection_score,evidence,temporal_confirmed,reference_available,
                         similarity=None,negative_similarity=None):
        if t!=self.last_t or self.state!=State.REDETECTION_REQUESTED:
            raise ValueError('candidate without a current causal detection request')
        if not evidence.valid():raise ValueError('invalid candidate evidence')
        self._transition(State.REID_CANDIDATE,t,'candidate_from_current_rgb')
        reasons=[]
        if not math.isfinite(detection_score) or detection_score<self.rules.minimum_detection_score:
            reasons.append('detector_score')
        if not self.usable(evidence):reasons.append('current_candidate_support')
        if not temporal_confirmed:reasons.append('missing_temporal_confirmation')
        if reference_available:
            if similarity is None or not math.isfinite(similarity) or similarity<self.rules.minimum_reid_similarity:
                reasons.append('reid_similarity')
            if negative_similarity is not None and (not math.isfinite(negative_similarity) or
                similarity is None or similarity-negative_similarity<self.rules.minimum_reid_margin):
                reasons.append('reid_hard_negative_margin')
        # Missing reference is explicit. There is no fabricated perfect cosine
        # and no future reference; independent detector/support evidence remains.
        if reasons:
            self.pending=None;self._transition(State.REANCHOR_REJECTED,t,','.join(reasons))
            return False
        self.pending=dict(timestep=t,reference_available=reference_available,similarity=similarity)
        return True

    def commit(self,t,current_validated):
        if self.pending is None or self.pending['timestep']!=t or t!=self.last_t:
            raise ValueError('no validated candidate transaction')
        self.pending=None
        if not current_validated:
            self._transition(State.REANCHOR_REJECTED,t,'fresh_memory_validation_failed_keep_original_memory')
            return False
        self.ever_found=True;self.currently_valid=True;self.last_accept=t
        self.bad_run=0;self.good_run=1
        self._transition(State.REANCHOR_ACCEPTED,t,'fresh_memory_current_validation_pass')
        return True

    def reject(self,t,reason):
        if t!=self.last_t or self.state not in (State.REDETECTION_REQUESTED,State.REID_CANDIDATE):
            raise ValueError('rejection without a current candidate request')
        self.pending=None;self._transition(State.REANCHOR_REJECTED,t,reason)
