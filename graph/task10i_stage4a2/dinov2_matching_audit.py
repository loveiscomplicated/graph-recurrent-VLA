"""Historical Development diagnostic. GT masks only in explicit oracle branches.

This is not a candidate or independent evaluation. Actor identity is scoped to
physical parent. Same-category objects in different parents are not negatives.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from graph.task10i_stage4a.run_pipeline import read, write, sha, STAGE3, CAMERAS


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def download_model(folder):
    """Public read-only download pinned to API-resolved commit, no credentials."""
    folder.mkdir(parents=True, exist_ok=True)
    identity_path = folder / 'download_identity.json'
    if identity_path.exists():
        receipt = read(identity_path)
        assert all(sha(folder / p) == h for p, h in receipt['files'].items())
        return receipt
    with urllib.request.urlopen('https://huggingface.co/api/models/facebook/dinov2-small', timeout=30) as response:
        info = json.load(response)
    revision = info['sha']
    files = ['config.json', 'preprocessor_config.json', 'model.safetensors']
    for name in files:
        url = f'https://huggingface.co/facebook/dinov2-small/resolve/{revision}/{name}'
        tmp = folder / (name + '.partial')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=120) as response, tmp.open('wb') as stream:
                    while chunk := response.read(1 << 20):
                        stream.write(chunk)
                tmp.replace(folder / name)
                break
            except Exception:
                if attempt == 2:
                    raise
    receipt = dict(model='facebook/dinov2-small', revision=revision,
                   files={p: sha(folder / p) for p in files}, downloaded_at=time.time())
    write(identity_path, receipt)
    return receipt


def crop_input(rgb, mask, padded):
    """Prediction-only crop with a separately resized support mask."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
    image = rgb[top:bottom, left:right].copy()
    support = mask[top:bottom, left:right].copy()
    if padded:
        h, w = support.shape
        side = max(h, w)
        canvas = np.full((side, side, 3), 127, np.uint8)
        sm = np.zeros((side, side), bool)
        y, x = (side-h)//2, (side-w)//2
        canvas[y:y+h, x:x+w] = image
        sm[y:y+h, x:x+w] = support
        image, support = canvas, sm
    return image, support, [left, top, right, bottom]


class Features:
    def __init__(self, folder):
        from transformers import AutoModel, AutoImageProcessor
        self.model = AutoModel.from_pretrained(folder, local_files_only=True).cuda().eval()
        self.processor = AutoImageProcessor.from_pretrained(folder, local_files_only=True)
        self.calls = []

    @torch.inference_mode()
    def extract(self, rgb, mask, mode):
        box = None
        if mode == 'legacy_full_patch448':
            image, support = rgb, mask
            x = torch.from_numpy(np.ascontiguousarray(image)).cuda().permute(2, 0, 1)[None].float()/255
            x = F.interpolate(x, (448, 448), mode='bilinear', align_corners=False)
            x = (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
        else:
            cropped = crop_input(rgb, mask, padded=mode == 'padded_crop_cls')
            if cropped is None:
                return None
            image, support, box = cropped
            x = self.processor(images=Image.fromarray(image), return_tensors='pt')['pixel_values'].cuda()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = self.model(pixel_values=x)
        torch.cuda.synchronize()
        if mode == 'legacy_full_patch448':
            h = output.last_hidden_state[0, 1:].float()
            g = int(h.shape[0]**.5)
            weights = F.interpolate(torch.as_tensor(support, device='cuda').float()[None,None],
                                    size=(g,g), mode='area').reshape(-1)
            if weights.sum() < .3:
                return None
            vector = (h*weights[:,None]).sum(0)/weights.sum()
        else:
            vector = output.last_hidden_state[0,0].float()
        vector = F.normalize(vector, dim=0)
        self.calls.append(dict(mode=mode, rgb_shape=list(rgb.shape), crop_box=box,
                          input_shape=list(x.shape), embedding_shape=list(vector.shape),
                          norm=float(vector.norm()), forward_ms=1000*(time.perf_counter()-start)))
        return vector.cpu().numpy()


def run(root, model_folder):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True, exist_ok=False)
    with ExperimentLogger(team='integration', model='Task10I_Stage4A2_DINOv2_diagnostic',
                          seed=42, root=root/'experiments', job_type='evaluation',
                          config=dict(partition='historical_development', independent_test=False)) as log:
        identity = download_model(model_folder)
        write(root/'dinov2_model_identity.json', identity)
        torch.set_num_threads(1)
        torch.manual_seed(42)
        extractor = Features(model_folder)
        write(root/'dinov2_processor_readback.json', extractor.processor.to_dict())
        audits = Path('/workspace/task10i_stage4a_completion_20261101_v1/run')
        entries = {str(e['episode_id']): e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
        by_parent = {}
        for path in sorted((audits/'corrected/perception').glob('*/receipt.json')):
            e = entries[path.parent.name]
            # One deterministic source per parent, no mask-quality selection.
            by_parent.setdefault(e['parent_id'], (e, read(path)))
        metadata, embeddings = [], []
        for parent, (entry, receipt) in sorted(by_parent.items()):
            eid = str(entry['episode_id'])
            ref_path = Path(receipt['corrected_reference'])
            assert sha(ref_path) == receipt['corrected_reference_sha256']
            reference = read(ref_path)
            masks_path = audits/'baseline/perception'/eid/'masks.npz'
            assert sha(masks_path) == receipt['masks_sha256']
            with np.load(masks_path) as f:
                predicted = np.unpackbits(f['mask_bits'], axis=1).reshape(-1,2,2,128,128).astype(bool)
            with np.load(ref_path.parent/'segmentation.npz') as f:
                exact, segmentation = f['exact_frame'].copy(), f['actor_segmentation'].copy()
            with h5py.File(entry['raw_path']) as raw:
                for t in sorted({0, entry['actions']//2, entry['actions']}):
                    if not exact[t]:
                        continue
                    for ci, camera in enumerate(CAMERAS):
                        rgb = raw[camera+'/rgb'][t]
                        for oid, name in enumerate(('peg', 'box', 'robot')):
                            target = np.isin(segmentation[t,ci], reference['robot_ids'] if name == 'robot' else reference['object_ids'][name])
                            branches = [('ORACLE_DIAGNOSTIC', target)]
                            if oid < 2:
                                branches.append(('PRACTICAL_PREDICTION', predicted[t,ci,oid]))
                            for branch, mask in branches:
                                if not mask.any():
                                    continue
                                for mode in ('legacy_full_patch448', 'bbox_crop_cls', 'padded_crop_cls'):
                                    embedding = extractor.extract(rgb, mask, mode)
                                    if embedding is None:
                                        continue
                                    # Pair labels follow observed physical Actor support,
                                    # not the semantic slot requested by the tracker.
                                    masses = {actor: int((mask & np.isin(segmentation[t,ci],
                                        reference['robot_ids'] if actor == 'robot' else reference['object_ids'][actor])).sum())
                                        for actor in ('peg','box','robot')}
                                    winner = max(masses, key=masses.get)
                                    actor_label = winner if masses[winner]/mask.sum() >= .8 else None
                                    metadata.append(dict(index=len(metadata), parent=parent, episode=eid,
                                        timestep=t, camera=camera, semantic=name, branch=branch, crop=mode,
                                        actual_actor=actor_label,
                                        actor_label_rule='>=80% GT support in physical parent; mixed/background excluded from labelled pairs',
                                        reference_kind='EXACT_GT', gt_purity=float((mask & target).sum()/mask.sum()),
                                        gt_visible_pixels=int(target.sum()), mask_pixels=int(mask.sum()),
                                        input_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
                                    embeddings.append(embedding)
            print('DINOv2 parent', parent, len(metadata), flush=True)
            write(root/'partial_provenance.json', metadata)
        vectors = np.stack(embeddings)
        np.savez_compressed(root/'embeddings.npz', embeddings=vectors)
        pairs = []
        for i, a in enumerate(metadata):
            for j in range(i+1, len(metadata)):
                b = metadata[j]
                if (a['parent'], a['branch'], a['crop']) != (b['parent'], b['branch'], b['crop']):
                    continue
                if a['actual_actor'] is None or b['actual_actor'] is None:
                    continue
                if a['actual_actor'] == b['actual_actor'] == 'robot':
                    continue
                positive = a['actual_actor'] == b['actual_actor']
                if positive and a['timestep'] == b['timestep'] and a['camera'] == b['camera']:
                    continue
                pairs.append(dict(a=i,b=j,parent=a['parent'],branch=a['branch'],crop=a['crop'],
                    positive=positive, cross_camera=a['camera'] != b['camera'],
                    similarity=float(vectors[i] @ vectors[j]),
                    hard_negative=not positive and 'robot' in (a['actual_actor'], b['actual_actor'])))
        metrics = []
        for branch in ('ORACLE_DIAGNOSTIC','PRACTICAL_PREDICTION'):
            for crop in ('legacy_full_patch448','bbox_crop_cls','padded_crop_cls'):
                rows = [r for r in pairs if r['branch'] == branch and r['crop'] == crop]
                for threshold in np.linspace(-1,1,201):
                    tp=sum(r['positive'] and r['similarity'] >= threshold for r in rows)
                    fp=sum(not r['positive'] and r['similarity'] >= threshold for r in rows)
                    pos=sum(r['positive'] for r in rows); neg=len(rows)-pos
                    metrics.append(dict(branch=branch,crop=crop,threshold=float(threshold),
                        positives=pos,negatives=neg,tpr=tp/pos if pos else None,
                        fpr=fp/neg if neg else None,precision=tp/(tp+fp) if tp+fp else None,
                        false_rejection_rate=1-tp/pos if pos else None))
        csv_write(root/'dinov2_positive_negative_pairs.csv', pairs)
        csv_write(root/'dinov2_reidentification_metrics.csv', metrics)
        write(root/'dinov2_reference_provenance.json', metadata)
        write(root/'dinov2_call_trace.json', extractor.calls)
        write(root/'scope.json',dict(status='HISTORICAL_DEVELOPMENT_DIAGNOSTIC_FINISHED',
              run_id=log.run_id, parents=len(by_parent), embeddings=len(metadata), pairs=len(pairs),
              physical_instance_identity='parent+semantic actor; cross-parent same category excluded',
              reference_update_policy_tested=False, causal_candidate_evaluated=False,
              mixed_background_pairs_excluded=True, full_background_hard_negative_benchmark_pending=True,
              independent_test=False, threshold_selected=False,
              source_sha256=sha(Path(__file__))))
        log.link_output(root/'scope.json',kind='metrics')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('model_folder', type=Path)
    args=p.parse_args()
    run(args.root,args.model_folder)
