"""Static diagnostic of executed non-Final occlusion capture and aligned GT."""
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
from PIL import Image,ImageDraw
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS


def run(root,inputs):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    config=dict(source_sha256=sha(Path(__file__)),engineering_only=True,final_data=False,
        inputs={str(p):sha(p) for p in inputs.glob('*.npz')})
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_capture_gallery',
            seed=42,root=root/'experiments',job_type='analysis',config=config) as logger:
        artifacts=[]
        for stratum in ('main','wrist','both'):
            receipt=read(inputs/(stratum+'_receipt.json'))
            image=Image.new('RGB',(4*256,2*290+28),(24,24,24));draw=ImageDraw.Draw(image)
            draw.text((8,8),stratum+' intended occluder; actual visible counts shown; green=Peg red=Box GT boundary',fill='white')
            with np.load(inputs/(stratum+'.npz')) as data:
                for ci,c in enumerate(CAMERAS):
                    for col,t in enumerate((7,8,19,20)):
                        rgb=data[f't{t}_{c}_rgb'].copy();seg=data[f't{t}_segmentation'][ci]
                        ids=receipt['rows'][t]['reference']['object_ids']
                        for name,color in [('peg',[50,255,80]),('box',[255,80,80])]:
                            mask=np.isin(seg,ids[name]);rgb[mask&~binary_erosion(mask)]=color
                        x,y=col*256,28+ci*290
                        image.paste(Image.fromarray(rgb).resize((256,256),Image.Resampling.NEAREST),(x,y))
                        count=receipt['rows'][t]['visible_pixels'][c]
                        draw.text((x+4,y+258),f'{c} t{t}; Peg={count["peg"]} Box={count["box"]}',fill='white')
            path=root/(stratum+'.png');image.save(path)
            with Image.open(path) as check:assert check.size==image.size;check.verify()
            artifacts.append(dict(path=path.name,sha256=sha(path)))
        write(root/'scope.json',dict(status='ENGINEERING_CAPTURE_GALLERY_COMPLETE',run_id=logger.run_id,
            artifacts=artifacts,source=config,final_data=False,GT_evaluation_only=True))
        logger.link_output(root/'scope.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('inputs',type=Path)
    a=p.parse_args();run(a.root,a.inputs)
