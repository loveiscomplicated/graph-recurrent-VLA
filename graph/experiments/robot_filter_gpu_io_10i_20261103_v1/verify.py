"""Saved-mask device/semantic regression. Never runs SAM2 or Graph builds."""
from pathlib import Path
import os,sys,json,hashlib,unittest,inspect
HERE=Path(__file__).resolve().parent
OVERLAY=HERE.parents[2]
OLD=Path('/workspace/task10i_robot_filter_optimization_20261103_v1/code_final_v4')
sys.path.insert(0,str(OLD))
# Reuse only the frozen input loader and baseline classifier context, not its
# execute()/Graph benchmark. No timings of other modules are collected.
from run import C,FROZEN,geometry,load_inputs,masks_for,Candidate,ExperimentLogger,np,torch
sys.path.insert(0,str(OVERLAY))
# The frozen source already imported its old tracker before the overlay was
# selected. Load the changed module in this isolated validation process only.
import importlib.util
spec=importlib.util.spec_from_file_location('visual.runtime.tracking',OVERLAY/'visual/runtime/tracking.py')
tracking=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tracking;spec.loader.exec_module(tracking)
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.test_gpu_io import NoReadback
from graph.persistent_geometry.fusion import filtered_masks
import graph.plan_a_geometry.core as core


def main():
    output=Path(os.environ['TASK10I_GPU_IO_OUTPUT']);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    cases=json.loads((OLD/'regression.json').read_text())['cases']+json.loads((OLD/'extra.json').read_text())['cases']
    full_masks=os.environ.get('TASK10I_GPU_VERIFY_FULL_MASKS')=='1'
    if full_masks:cases=[dict(scene='10i_s1_0924_t080',size=size,mask_mode='full') for size in (128,512,768,1024)]
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    config=dict(cases=cases,full_mask_stress=full_masks,sam2_execution=False,graph_execution=False,training=False,original_processes_resumed=False,
        boundary='CUDA SAM2 mask tensor -> CUDA clean mask; CPU sensor/FK inputs may upload, no mask readback',
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
        sam2_predict_source=inspect.getsource(SAM2ImagePredictor.predict),
        source_hashes={str(p.relative_to(OVERLAY)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OVERLAY.rglob('*.py')})
    C.write(output/'config.json',config)
    suite=unittest.defaultTestLoader.loadTestsFromNames(['graph.experiments.robot_filter_gpu_io_10i_20261103_v1.test_gpu_io'])
    specs={s['id']:s for s in C.read(FROZEN/'protocol.json')['scenes']};rows=[]
    with ExperimentLogger(team='integration',model='task10i_robot_filter_gpu_io',job_type='evaluate',seed=42,root=output/'experiments',config=config) as log:
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise AssertionError('device contract tests failed')
        for i,case in enumerate(cases):
            geo=geometry();inputs=load_inputs(case['scene'],case['size'],geo)
            masks=masks_for(specs[case['scene']],case['size'],0,'actual')
            if full_masks:masks={k:np.stack((np.ones_like(v[0]),np.zeros_like(v[1]))) for k,v in masks.items()}
            device_masks={k:torch.as_tensor(v,device='cuda') for k,v in masks.items()}
            before={k:v.clone() for k,v in device_masks.items()}
            filt=CudaRobotFilter()
            batch=filt.filter_cameras(inputs[0],inputs[2],device_masks,diagnostics=True)
            filt.filter_cameras(inputs[0],inputs[2],device_masks)
            with NoReadback():clean_batch=filt.filter_cameras(inputs[0],inputs[2],device_masks)
            for camera in inputs[0]:
                with Candidate(9):reference=core.classify_robot(camera,inputs[2])
                result=batch[camera.camera]
                good=result['evaluated_pixels'].cpu().numpy()
                for field in ('confirmed_robot','uncertain','candidate','link_owner','inside_known_link_volume'):
                    np.testing.assert_array_equal(result[field].cpu().numpy()[good],reference[field][good],err_msg=field)
                a=result['predicted_robot_depth'].cpu().numpy()[good];b=reference['predicted_robot_depth'][good]
                np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
                expected=filtered_masks(masks[camera.camera],reference)[0]
                np.testing.assert_array_equal(result['clean_masks'].cpu().numpy(),expected)
                # Input creation/verification readback stays outside this guard.
                clean=clean_batch[camera.camera];raw=device_masks[camera.camera]
                assert clean.is_cuda and clean.dtype==torch.bool and clean.shape==raw.shape
                assert torch.equal(clean,result['clean_masks']) and torch.equal(before[camera.camera],raw)
                finite=np.isfinite(a)&np.isfinite(b)
                rows.append(dict(case=case,camera=camera.camera,mask_sha256=C.array_sha(masks[camera.camera]),
                    clean_sha256=C.array_sha(expected),clean_exact=True,owner_and_pixel_flags_exact=True,
                    depth_max_abs=float(np.max(np.abs(a[finite]-b[finite]),initial=0)),
                    cuda_input_output=True,no_device_readback=True))
            C.write(output/'results.json',dict(status='PARTIAL',completed=i+1,expected=len(cases),rows=rows))
            print(json.dumps(dict(completed=i+1,total=len(cases),case=case,status='PASS')),flush=True)
            log.log(dict(cases_complete=i+1),step=i+1)
        original={}
        for pid in (327662,327663,327664,327665,327666,357110,357111):
            lines=Path(f'/proc/{pid}/status').read_text().splitlines()
            state=next(x for x in lines if x.startswith('State:'));assert '\tT' in state
            original[str(pid)]=state
        C.write(output/'results.json',dict(status='PASS',completed=len(cases),rows=rows,original_processes=original))
        log.link_output(output/'results.json');log.link_output(output/'config.json')


if __name__=='__main__':main()
