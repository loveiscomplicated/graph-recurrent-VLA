"""Evaluation-only geometry references and explicit per-object denominators."""
import numpy as np
from scipy.spatial import cKDTree,ConvexHull
from scipy.spatial.transform import Rotation

from visual.runtime.vision_packet import pose_matrix
from .contracts import transform


def sample_meshes(path, samples_per_object=30000):
    """Deterministic area-weighted GT surface reference, evaluation only.

    Distances are to sampled mesh surfaces; the sampling floor is reported and
    is not presented as exact analytic mesh distance.
    """
    rng = np.random.default_rng(42)
    result = {}
    with np.load(path, allow_pickle=False) as a:
        for oid in (0, 1):
            triangles = []
            outward=[];hulls=[]
            for key in a.files:
                prefix = f'object{oid}_vertices'
                if key.startswith(prefix):
                    index = key[len(prefix):]
                    vertices=a[key]
                    tri=vertices[a[f'object{oid}_faces{index}']]
                    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                    sign=np.sign(np.sum(cross*(tri.mean(1)-vertices.mean(0)),axis=1))
                    sign[sign==0]=1
                    triangles.append(tri);outward.append(cross*sign[:,None])
                    hulls.append(ConvexHull(vertices).equations)
            triangles = np.concatenate(triangles)
            cross = np.concatenate(outward)
            areas = np.linalg.norm(cross, axis=1)
            choices = rng.choice(len(triangles), samples_per_object, p=areas/areas.sum())
            r = rng.random((samples_per_object, 2))
            root = np.sqrt(r[:, :1])
            bary = np.c_[1-root[:, 0], root[:, 0]*(1-r[:, 1]), root[:, 0]*r[:, 1]]
            points = np.einsum('ni,nij->nj', bary, triangles[choices])
            normals = cross[choices]/areas[choices, None]
            # Adjacent collision boxes contain internal interface faces. Such
            # faces are not the rigid object's external surface. Remove samples
            # whose outward displacement still lies in any collision component.
            outside=points+normals*.0002
            internal=np.zeros(len(points),bool)
            for equations in hulls:
                internal |= np.all(outside@equations[:,:3].T+equations[:,3]<=1e-10,axis=1)
            points,normals=points[~internal],normals[~internal]
            result[oid] = dict(points=points, normals=normals, tree=cKDTree(points))
    return result


def filter_metrics(camera, masks, result, gt, *, finger_world=None):
    segmentation = gt[camera.camera+'/segmentation']
    robot = np.isin(segmentation, gt['robot_ids']) & np.isfinite(camera.depth)
    removed = result['confirmed_robot']
    uncertain = result['uncertain']
    object_masks = [segmentation == actor for actor in gt['actor_ids']]
    rows = []
    for oid, correct in enumerate(object_masks):
        correct &= np.isfinite(camera.depth)
        predicted = masks[oid] & np.isfinite(camera.depth)
        remaining = predicted & ~removed & ~uncertain & ~(masks[0] & masks[1])
        depth_near = correct & np.isfinite(result['predicted_robot_depth']) & (np.abs(camera.depth-result['predicted_robot_depth']) < .01)
        near=np.zeros(correct.shape,bool)
        if finger_world is not None and correct.any():
            points=camera.world_points()[correct]
            distance=np.linalg.norm(points[:,None]-np.asarray(finger_world)[None],axis=-1).min(1)
            near[correct]=distance<.02
        def ratio(num, den):
            return float(num/den) if den else None
        rows.append(dict(camera=camera.camera, object_id=oid,
            robot_visible=int(robot.sum()), robot_removed=int((removed & robot).sum()),
            robot_removal_recall=ratio((removed & robot).sum(), robot.sum()),
            gt_object_visible=int(correct.sum()), object_preservation=ratio((correct & ~removed & ~uncertain).sum(), correct.sum()),
            confirmed_false_removal=ratio((correct & removed).sum(), correct.sum()),
            uncertain_object_fraction=ratio((correct & uncertain).sum(), correct.sum()),
            predicted_before=int(predicted.sum()), predicted_after=int(remaining.sum()),
            robot_contamination_before=ratio((predicted & robot).sum(), predicted.sum()),
            robot_contamination_after=ratio((remaining & robot).sum(), remaining.sum()),
            object_purity_before=ratio((predicted & correct).sum(), predicted.sum()),
            object_purity_after=ratio((remaining & correct).sum(), remaining.sum()),
            contact_near_pixels=int(near.sum()), contact_near_false_removal=ratio((near & removed).sum(), near.sum()),
            contact_near_definition='within20mm of an observed finger-pad collision centre; geometry proximity, not a force/contact claim',
            depth_near_robot_pixels=int(depth_near.sum()),depth_near_robot_false_removal=ratio((depth_near & removed).sum(),depth_near.sum()),
            object_missing_before=bool(correct.any() and not (predicted & correct).any()),
            object_missing_after=bool(correct.any() and not (remaining & correct).any())))
    return rows


def visible_reference(cameras, gt, names=None):
    result = {0:[], 1:[]}
    for camera in cameras:
        if names is not None and camera.camera not in names:
            continue
        world = camera.world_points()
        for oid, actor in enumerate(gt['actor_ids']):
            selected = (gt[camera.camera+'/segmentation'] == actor) & np.isfinite(camera.depth)
            result[oid].append(world[selected])
    return {oid:np.concatenate(rows).reshape(-1, 3) if rows else np.empty((0, 3)) for oid, rows in result.items()}


def geometry_summary_numpy(points):
    points = np.asarray(points)
    if not len(points):
        return np.zeros(15)
    lo, hi = points.min(0), points.max(0)
    return np.concatenate((points.mean(0), lo, hi, (lo+hi)/2, hi-lo))


def provenance_metrics(variant,records,history,cache,timestep):
    """First and fused source-actor purity; evaluation labels never return to maps."""
    result={}
    for oid in (0,1):
        points=[r for r in records if r['object_id']==oid and not r.get('removed',False)]
        first=[];totals=np.zeros(4,np.int64)
        for point in points:
            key=(variant,point['persistent_node_id'])
            entry=cache.setdefault(key,dict(processed=0,counts=np.zeros(4,np.int64),first=None))
            sources=point['source_observation_ids']
            if entry['processed']>len(sources):raise ValueError('source history shrank for persistent identity')
            for _,source_t,camera,pixel in sources[entry['processed']:]:
                if source_t>timestep:raise ValueError('future source in geometry evaluation')
                old=history[source_t]
                actor=int(old[camera+'/segmentation'].ravel()[pixel])
                label=(0 if actor==old['actor_ids'][oid] else 1 if actor in old['robot_ids']
                       else 3 if actor==old['actor_ids'][1-oid] else 2)
                entry['counts'][label]+=1
                if entry['first'] is None:entry['first']=label
            entry['processed']=len(sources);first.append(entry['first']);totals+=entry['counts']
        labels=np.array(first)
        result[oid]=dict(source_object_purity=float((labels==0).mean()) if len(labels) else None,
            source_robot_contamination=float((labels==1).mean()) if len(labels) else None,
            source_background_contamination=float((labels==2).mean()) if len(labels) else None,
            source_wrong_object_contamination=float((labels==3).mean()) if len(labels) else None,
            source_background_or_wrong_object=float((labels>=2).mean()) if len(labels) else None,
            fused_source_object_purity=float(totals[0]/totals.sum()) if totals.sum() else None,
            fused_source_observations=int(totals.sum()),source_purity_definition='first source per node; fused purity additionally counts all supporting measurements')
    return result


class GeometryEvaluator:
    def __init__(self, meshes):
        self.meshes = meshes
        self.cumulative = {name:{oid:{} for oid in (0, 1)} for name in ('main','both')}
        self.previous = {}
        self.initial_pose = {}
        self.first_node_local={}
        self.before_current={}

    def references(self, cameras, gt, main):
        refs = {'main':visible_reference(cameras, gt, [main]), 'both':visible_reference(cameras, gt)}
        self.before_current={name:{oid:np.array(list(points.values())).reshape(-1,3)
            for oid,points in objects.items()} for name,objects in self.cumulative.items()}
        for name, objects in refs.items():
            for oid, world in objects.items():
                local = transform(world, np.linalg.inv(pose_matrix(gt['actor_pose'][oid])))
                for key, point in zip(np.floor(local/.002).astype(int), local):
                    self.cumulative[name][oid].setdefault(tuple(key), point.copy())
        return refs

    def measure(self, variant, records, poses, gt, references, timestep, *, budget_mode='native_variable_N'):
        rows = []
        allowed = 'main' if variant in ('G0','G1','G2') else 'both'
        identity=variant+'@'+budget_mode
        previous = self.previous.get(identity)
        for oid in (0, 1):
            selected = [r for r in records if r['object_id'] == oid and not r.get('removed',False)]
            world = np.array([r['world_xyz'] for r in selected]).reshape(-1, 3)
            actor = pose_matrix(gt['actor_pose'][oid])
            local = transform(world, np.linalg.inv(actor))
            distance, index = self.meshes[oid]['tree'].query(local)
            visible = references[allowed][oid]
            cumulative_local = np.array(list(self.cumulative[allowed][oid].values())).reshape(-1, 3)
            cumulative_world = transform(cumulative_local, actor)
            common_local=np.array(list(self.cumulative['both'][oid].values())).reshape(-1,3)
            common_world=transform(common_local,actor)
            tree = cKDTree(world) if len(world) else None
            def coverage(points):
                if not len(points):return None
                return float((tree.query(points)[0]<.004).mean()) if tree is not None else 0.
            normal = np.array([r['world_normal'] for r in selected]).reshape(-1, 3)
            normal_valid = np.array([r['normal_valid'] for r in selected], bool)
            gt_normal = self.meshes[oid]['normals'][index] @ actor[:3, :3].T
            cosine = np.abs(np.sum(normal*gt_normal, axis=1))
            count = len(selected)
            estimated = np.array([r['estimated_only'] for r in selected], bool)
            confidence = np.array([r['confidence']*r['pose_confidence'] for r in selected])
            uncertainty = np.array([r['position_uncertainty'] for r in selected])
            row = dict(variant=variant, budget_mode=budget_mode, object_id=oid, timestep=timestep, count=count,
                currently_observed=int((~estimated).sum()), persistent_estimated=int(estimated.sum()),
                surface_precision_4mm=float((distance < .004).mean()) if count else None,
                surface_distance_mean_mm=float(distance.mean()*1000) if count else None,
                surface_distance_p95_mm=float(np.percentile(distance,95)*1000) if count else None,
                current_visible_coverage_4mm=coverage(visible), cumulative_observed_coverage_4mm=coverage(cumulative_world),
                common_both_visible_coverage_4mm=coverage(references['both'][oid]),
                common_union_observed_coverage_4mm=coverage(common_world),
                common_both_visible_reference=len(references['both'][oid]),common_union_observed_reference=len(common_world),
                current_visible_reference=len(visible), cumulative_observed_reference=len(cumulative_world),
                estimated_correct=int((estimated & (distance < .004)).sum()),
                ghost_count=int((estimated & (distance > .008)).sum()),
                normal_consistency=float(cosine[normal_valid].mean()) if normal_valid.any() else None,
                confidence_mean=float(confidence.mean()) if count else None,
                uncertainty_mean_mm=float(uncertainty.mean()*1000) if count else None,
                newly_added=sum(r['first_seen_timestep'] == timestep for r in selected),
                newly_added_correct=sum(r['first_seen_timestep'] == timestep and d < .004 for r,d in zip(selected,distance)),
                input_sensor_scope=allowed,
                distance_reference='deterministic area-weighted external boundary samples of the collision-shape union; internal component faces excluded',
                mesh_reference_samples=len(self.meshes[oid]['points']))
            drift=[]
            for point,position in zip(selected,local):
                node_key=(identity,oid,point['persistent_node_id'])
                if node_key not in self.first_node_local:self.first_node_local[node_key]=position.copy()
                drift.append(np.linalg.norm(position-self.first_node_local[node_key]))
            row['object_local_surface_drift_mean_mm']=(float(np.mean(drift)*1000) if drift
                and variant in ('G2','G4','O1','O2','O3') else None)
            new=np.array([r['first_seen_timestep']==timestep for r in selected],bool)
            past=self.before_current.get(allowed,{}).get(oid,np.empty((0,3)))
            previous_distance=cKDTree(past).query(local)[0] if len(past) else np.full(count,np.inf)
            row['new_correct_previously_unseen_nodes']=int((new&(distance<.004)&(previous_distance>.004)).sum())
            row['new_correct_previously_observed_nodes']=int((new&(distance<.004)&(previous_distance<=.004)).sum())
            if count:
                row['duplicate_neighbour_fraction_1mm'] = float((cKDTree(local).query(local,k=2)[0][:,1] < .001).mean()) if count>1 else 0.
            for label, low, high in (('low',0,.25),('medium',.25,.6),('high',.6,1.01)):
                bins = (confidence >= low) & (confidence < high)
                row[f'{label}_confidence_count'] = int(bins.sum())
                row[f'{label}_confidence_error_mm'] = float(distance[bins].mean()*1000) if bins.any() else None
            key = (identity, oid)
            if poses is not None and count:
                pose = poses[oid]
                if key not in self.initial_pose:
                    self.initial_pose[key] = (actor.copy(), pose.copy())
                first_actor, first_map = self.initial_pose[key]
                expected = actor @ np.linalg.inv(first_actor) @ first_map
                row['pose_translation_error_mm'] = float(np.linalg.norm(pose[:3,3]-expected[:3,3])*1000)
                row['pose_rotation_error_deg'] = float(np.rad2deg(Rotation.from_matrix(pose[:3,:3] @ expected[:3,:3].T).magnitude()))
            if previous is not None:
                old_records, old_gt = previous
                old = {r['persistent_node_id']:r for r in old_records if r['object_id']==oid}
                motion = actor @ np.linalg.inv(pose_matrix(old_gt['actor_pose'][oid]))
                error = [np.linalg.norm(r['world_xyz']-transform(old[r['persistent_node_id']]['world_xyz'], motion))
                         for r in selected if r['persistent_node_id'] in old]
                row['motion_update_error_mm'] = float(np.mean(error)*1000) if error else None
                row['retained_previous_nodes'] = len(error)
                row['previous_active_nodes'] = len(old)
            rows.append(row)
        # Snapshot copies are essential: later surfel fusion must not rewrite
        # the evaluator's previous physical timestep.
        self.previous[identity] = ([{k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in r.items()} for r in records],
                                  {'actor_pose':gt['actor_pose'].copy()})
        return rows
