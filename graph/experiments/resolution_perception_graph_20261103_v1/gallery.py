"""Render saved native RGB, frozen r0 masks and true Graph tensor coordinates."""
from common import *
import numpy as np
from PIL import Image
from visual.runtime.vision_packet import pose_matrix
from graph.persistent_geometry.contracts import transform


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    predictions={r['id']:r for r in read(ROOT/'api_predictions.json')['rows']}
    from graph_worker import resolve_folder
    graphs=read(ROOT/'graph_index_actual.json')['rows']
    fig=plt.figure(figsize=(16,11),layout='constrained')
    grid=fig.add_gridspec(3,4);clouds=[];axes=[]
    for col,size in enumerate(SIZES):
        for row,camera in enumerate(CAMERAS):
            ax=fig.add_subplot(grid[row,col]);axes.append(ax)
            iid=image_id('seed98202_t000',size,camera);p=predictions[job_id(iid,0)]
            image=np.asarray(Image.open(ROOT/'sensors/seed98202_t000'/str(size)/(camera+'.png')))
            ax.imshow(image)
            with np.load(ROOT/'masks/actual/static'/p['id']/'masks.npz',allow_pickle=False) as z:masks=z['masks']
            for oid,(name,color) in enumerate((('peg','#ff3d3d'),('box','#d79b39'))):
                if masks[oid].any() and not masks[oid].all():ax.contour(masks[oid],levels=[.5],colors=[color],linewidths=1)
                box=p['parsed'][name]['pixels']
                if box:
                    x0,y0,x1,y1=box;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color=color,lw=1))
            ax.set_title(f'{camera} / {size} / first Terra response',fontsize=10);ax.axis('off')
        ax=fig.add_subplot(grid[2,col],projection='3d');axes.append(ax)
        outcome=next(r for r in graphs if r['scene']=='seed98202_t000' and r['size']==size and r['repeat']==0)
        folder=resolve_folder(outcome) if outcome['status']=='COMPLETE' else ROOT/outcome['folder']
        if not (folder/'graph.npz').exists():ax.text2D(.05,.5,'Graph unavailable',transform=ax.transAxes);continue
        with np.load(folder/'graph.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in FIELDS}
        with np.load(ROOT/'sensors/seed98202_t000'/str(size)/'robot.npz',allow_pickle=False) as z:m=pose_matrix(z['tcp_pose'])
        world=transform(a['node_input'][:,:3],m);clouds.append(world[3:])
        for kind,color,label in ((3,'#ef4444','Peg'),(4,'#b7791f','Box')):
            take=a['node_type']==kind
            ax.scatter(*world[take].T,s=8,c=color,label=label)
        for i,j in a['edge_index'].T:
            if i<j and i>=3 and j>=3:
                line=world[[i,j]];ax.plot(*line.T,color='gray',alpha=.18,lw=.35)
        ax.view_init(elev=24,azim=-55);ax.set_title(f'Predicted Graph / {size}',fontsize=10)
        ax.set_xlabel('world X m',fontsize=8);ax.set_ylabel('Y m',fontsize=8);ax.set_zlabel('Z m',fontsize=8)
    if clouds:
        xyz=np.concatenate(clouds);lo=xyz.min(axis=0);hi=xyz.max(axis=0);pad=np.maximum((hi-lo)*.1,.01)
        for ax in axes:
            if hasattr(ax,'set_zlim'):
                ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1]);ax.set_zlim(lo[2]-pad[2],hi[2]+pad[2])
    fig.suptitle('seed98202 t0 | frozen first response | identical world plot limits',fontsize=14)
    fig.savefig(ROOT/'seed98202_comparison.png',dpi=160);plt.close(fig)
    print('SEED_GALLERY_RENDERED',flush=True)


if __name__=='__main__':main()
