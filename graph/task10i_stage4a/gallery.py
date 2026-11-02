"""Evidence gallery from hash-checked raw, baseline cache and exact replay IDs."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from PIL import Image,ImageDraw
from .run_pipeline import read,sha,write

COLORS=np.array([[240,65,65],[50,160,255]])

def overlay(rgb,masks):
    image=rgb.astype(float).copy()
    for oid,m in enumerate(masks):image[m]=.5*image[m]+.5*COLORS[oid]
    return Image.fromarray(image.astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST)

def run(raw,cache,reference,output):
    identity=read(reference/'receipt.json');assert sha(raw)==identity['raw_sha256']
    assert sha(cache)=='a655de1f3916e2b3cb2af4a33bae7b8090eac102d0d4bfc90eb8578570eef2d1'
    output.mkdir(parents=True,exist_ok=True);frames=[0,17,54,87];audit=[]
    with h5py.File(raw) as f,np.load(cache,allow_pickle=False) as baseline,np.load(reference/'segmentation.npz') as gt:
        for ci,camera in enumerate(('base_camera','hand_camera')):
            panel=Image.new('RGB',(1024,4*286+28),(25,25,25));draw=ImageDraw.Draw(panel)
            for j,title in enumerate(('Raw RGB','EXACT_GT (red Peg / blue Box)','Causal SAM2','Final graph source pixels')):draw.text((j*256+5,7),title,fill='white')
            for row,t in enumerate(frames):
                assert gt['exact_frame'][t]
                rgb=f[camera+'/rgb'][t];seg=gt['actor_segmentation'][t,ci]
                truth=np.stack([np.isin(seg,identity['object_ids'][n]) for n in ('peg','box')])
                predicted=np.unpackbits(baseline['mask_bits'][t]).reshape(2,2,128,128)[ci].astype(bool)
                final=np.zeros((2,128,128),bool);point=baseline['source_point'];belongs=(baseline['point_frame'][point]==t)&(baseline['source_camera']==ci)
                for oid in (0,1):
                    rows=belongs&(baseline['node_type'][baseline['point_graph_node'][point]]==3+oid)
                    final[oid].ravel()[baseline['source_pixel'][rows]]=True
                images=[Image.fromarray(rgb).resize((256,256),Image.Resampling.NEAREST),overlay(rgb,truth),overlay(rgb,predicted),overlay(rgb,final)]
                for col,im in enumerate(images):panel.paste(im,(col*256,28+row*286))
                counts=dict(timestep=t,camera=camera,GT=truth.sum((1,2)).tolist(),SAM2=predicted.sum((1,2)).tolist(),graph_sources=final.sum((1,2)).tolist())
                audit.append(counts)
                for col,text in enumerate((f't={t} {camera}',f'GT Peg/Box: {counts["GT"]}',f'SAM2 Peg/Box: {counts["SAM2"]}',f'Graph sources: {counts["graph_sources"]}')):draw.text((col*256+4,28+row*286+260),text,fill='white')
            panel.save(output/(camera+'.png'))
    write(output/'gallery_identity.json',dict(raw_sha256=sha(raw),cache_sha256=sha(cache),reference_sha256=sha(reference/'segmentation.npz'),frames=frames,all_frames_exact=True,rows=audit))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('cache',type=Path);p.add_argument('reference',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.raw,a.cache,a.reference,a.output)
