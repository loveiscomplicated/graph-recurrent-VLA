"""Remote CUDA-only stage; accepts a GT-free transfer directory, never API keys."""
import argparse
from common import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    args = ap.parse_args()
    r = Path(args.root)
    receipt = r / 'sam2_predictions.json'
    if receipt.exists():
        raise SystemExit('SAM2 already completed; refusing a second load')
    assert not list(r.glob('.env*'))
    assert not (r / 'sam2_started.json').exists(), 'Refusing a second SAM2 execution'
    bundle = read(r / 'segment_manifest.json')
    assert sha(r / 'predictions.json') == bundle['predictions_sha256']
    assert sha(SAM_CHECKPOINT) == SAM_SHA
    import resource
    import time
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(4)
    torch.manual_seed(42)
    assert torch.cuda.is_available()
    write(r / 'sam2_started.json', {'started_utc':now(), 'SAM2_loads':1})
    start = time.perf_counter()
    model = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml', SAM_CHECKPOINT, device='cuda').eval()
    assert all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in model.parameters())
    predictor = SAM2ImagePredictor(model)
    torch.cuda.synchronize()
    cost = dict(load_ms=(time.perf_counter()-start)*1000, SAM2_loads=1, encoding=[], prompts=[],
                checkpoint_sha256=sha(SAM_CHECKPOINT), torch_version=torch.__version__,
                device=torch.cuda.get_device_name(), installed_defaults={
                    'mask_threshold': predictor.mask_threshold,
                    'max_hole_area': predictor._transforms.max_hole_area,
                    'max_sprinkle_area': predictor._transforms.max_sprinkle_area,
                    'dynamic_multimask_via_stability': model.sam_mask_decoder.dynamic_multimask_via_stability})
    rows = []
    predictions = read(r / 'predictions.json')['rows']
    for i, image in enumerate(bundle['images']):
        path = r / 'images' / (image['id'] + '.png')
        assert sha(path) == image['image_sha256']
        rgb = np.asarray(Image.open(path).convert('RGB'))
        assert rgb.shape == (512,512,3)
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            predictor.set_image(rgb)
        torch.cuda.synchronize()
        cost['encoding'].append({'image_id':image['id'], 'ms':(time.perf_counter()-t)*1000})
        for prediction in [p for p in predictions if p['image_id'] == image['id']]:
            for obj in ['peg','box']:
                box = prediction['parsed'][obj]
                if box['status'] != 'VALID':
                    continue
                t = time.perf_counter()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    masks, scores, _ = predictor.predict(box=np.asarray(box['pixels'],np.float32),
                                                         multimask_output=False, return_logits=True)
                torch.cuda.synchronize()
                mask = masks[0] > 0
                name = prediction['provider'] + '__' + image['id'] + '__' + obj + '.png'
                path = r / 'masks' / name
                path.parent.mkdir(exist_ok=True)
                Image.fromarray(mask.astype(np.uint8)*255).save(path)
                rows.append(dict(provider=prediction['provider'], image_id=image['id'], object=obj,
                    mask_file='masks/'+name, mask_sha256=sha(path), pixels=int(mask.sum()), quality=float(scores[0])))
                cost['prompts'].append(dict(id=name, ms=(time.perf_counter()-t)*1000))
        cost['peak_allocated_bytes'] = max(cost.get('peak_allocated_bytes',0),torch.cuda.max_memory_allocated())
        predictor.reset_predictor()
        write(r / 'sam2_progress.json',dict(completed_images=i+1,masks=len(rows)))
        write(receipt,dict(rows=rows,status='PARTIAL',completed_images=i+1,
                          predictions_sha256=sha(r / 'predictions.json')))
        write(r / 'sam2_latency_vram.json',cost)
        print('SAM2_PROGRESS',i+1, '/60 masks',len(rows),flush=True)
    cost['CPU_peak_ru_maxrss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(r / 'sam2_latency_vram.json',cost)
    write(receipt,dict(rows=rows,status='COMPLETE', frozen_utc=now(), predictions_sha256=sha(r / 'predictions.json'),
                       checkpoint_sha256=sha(SAM_CHECKPOINT)))
    print('SAM2_MASKS_FROZEN',len(rows),sha(receipt),flush=True)

if __name__ == '__main__': main()
