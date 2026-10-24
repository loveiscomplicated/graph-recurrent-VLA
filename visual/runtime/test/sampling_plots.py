"""Fixed-view comparison plots; colours are debugging only, GT is evaluation only."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from visual.runtime.rgbd import PEG, BOX
from visual.runtime.test.run_normal_comparison import bounds
from visual.runtime.sampling import extract_candidates, SamplingConfig

COLORS = dict(candidate='#ff5d41',depth='#cc54e5',priority='#ffff00',surface='#00d9ff',gt='#ff65cc',mask='#69ff72')


def overlays(output, frame, contexts, samples, evidence):
    context = contexts
    union=frame.masks[PEG]|frame.masks[BOX]
    views=[('Full RGB',None,(0,frame.depth.shape[1]-1,frame.depth.shape[0]-1,0)),
           ('Peg visible seam',PEG,bounds(context['regions']['peg']['gt_visible'],union,8)),
           ('Box physical rim',BOX,bounds(context['regions']['box']['hole_rim'],context['regions']['box']['gt_visible'],12))]
    fig,axes=plt.subplots(3,6,figsize=(21,12),constrained_layout=True)
    for axs,(title,oid,extent) in zip(axes,views):
        for col,ax in enumerate(axs):
            ax.imshow(frame.rgb)
            for name,i in [('peg',PEG),('box',BOX)]:
                if oid is not None and oid!=i:continue
                m=frame.masks[i]
                if m.any() and not m.all(): ax.contour(m,[.5],colors=[COLORS['mask']],linewidths=.45)
                uv=context['references'][name][0]
                ax.scatter(uv[:,0],uv[:,1],s=14,c=COLORS['gt'],marker='+',linewidths=.5)
            if col==0:
                ax.set_title(title+' | GT reference + masks',fontsize=10)
            else:
                method='S'+str(col-1); sample=samples[method]
                candidates=extract_candidates(evidence[method],SamplingConfig(),method)
                for i in (PEG,BOX) if oid is None else (oid,):
                    c=candidates[i]; h,w=frame.depth.shape
                    for key,color,marker in [('orientation_candidate',COLORS['candidate'],'.'),('depth_only_candidate',COLORS['depth'],'s')]:
                        # Depth evidence is shown independently in every method.
                        v,u=np.unravel_index(c.flat_index[c.metadata[key]],frame.depth.shape)
                        if key=='depth_only_candidate':ax.scatter(u,v,s=8,facecolors='none',edgecolors=color,marker=marker,linewidths=.3,alpha=.7)
                        else:ax.scatter(u,v,s=2,c=color,marker=marker,alpha=.6)
                keep=np.ones(len(sample.points),bool) if oid is None else sample.object_id==oid
                for priority,color,marker in [(False,COLORS['surface'],'o'),(True,COLORS['priority'],'*')]:
                    v,u=sample.pixels_vu[keep & (sample.priority_selected==priority)].T
                    ax.scatter(u,v,s=14 if priority else 8,c=color,marker=marker,edgecolors='black',linewidths=.2)
                ax.set_title(f'{method} | priority={int((sample.priority_selected&keep).sum())} / FPS={int((~sample.priority_selected&keep).sum())}',fontsize=10)
            ax.set(xlim=extent[:2],ylim=extent[2:]);ax.set_aspect('equal');ax.axis('off')
    handles=[Line2D([],[],marker=m,color=c,ls='',label=k) for k,c,m in [
        ('orientation candidate',COLORS['candidate'],'.'),('depth-only evidence',COLORS['depth'],'s'),
        ('priority selected',COLORS['priority'],'*'),('Surface FPS selected',COLORS['surface'],'o'),
        ('GT reference (eval only)',COLORS['gt'],'+'),('tracker mask',COLORS['mask'],'_')]]
    fig.legend(handles=handles,loc='outside lower center',ncol=6,fontsize=9)
    fig.suptitle(f'Source {frame.source_frame}: 256 total nodes, ratio .25, angle >20 deg, seed 0; identical crops\n'
                 'Depth is separate evidence; S4 permits at most 16 depth-only priority slots. Markers may overlap.')
    fig.savefig(output/f'f{frame.source_frame:04d}_overlay.png',dpi=140);plt.close(fig)


def clouds(output,frame,context,samples,evidence):
    fig=plt.figure(figsize=(21,11),layout='constrained')
    fig.set_constrained_layout_pads(h_pad=.18, w_pad=.04, hspace=.06, wspace=.02)
    for row,(name,oid) in enumerate([('peg',PEG),('box',BOX)]):
        cloud=context['surfaces'][name]['xyz']
        # Identical per-object viewport, not input filtering. Full coordinates are saved.
        if len(cloud):
            lo,hi=np.percentile(cloud,[.5,99.5],axis=0); centre=(lo+hi)/2
            radius=max(float(np.max(hi-lo))/2,.005)
        else:centre=np.array([0,0,.3]);radius=.05
        for col in range(6):
            ax=fig.add_subplot(2,6,row*6+col+1,projection='3d')
            if len(cloud):ax.scatter(*cloud.T,s=1.2 if col==0 else .4,c='#475566' if col==0 else '#999999',alpha=.85 if col==0 else .18,rasterized=True)
            if col:
                method='S'+str(col-1);sample=samples[method];keep=sample.object_id==oid
                for priority,color in [(False,'#008cbd'),(True,'#ff9b00')]:
                    pts=sample.points[keep&(sample.priority_selected==priority)]
                    ax.scatter(*pts.T,s=9 if priority else 5,c=color,depthshade=False)
                title=f'{name} {method}: {int(keep.sum())} selected'
            else:title=f'{name}: {len(cloud)} observed'
            if len(cloud):
                clipped=int((np.abs(cloud-centre)>radius).any(axis=1).sum())
                title+=f'\nviewport excludes {clipped} points'
            ax.set(xlim=(centre[0]-radius,centre[0]+radius),ylim=(centre[1]-radius,centre[1]+radius),zlim=(centre[2]-radius,centre[2]+radius))
            ax.view_init(elev=-65,azim=-85);ax.set_box_aspect((1,1,1));ax.set_title(title,fontsize=9,pad=12)
            ax.set_xlabel('CV X (m)',fontsize=7,labelpad=0);ax.set_ylabel('CV Y (m)',fontsize=7,labelpad=0);ax.set_zlabel('CV Z (m)',fontsize=7,labelpad=0)
            ax.tick_params(labelsize=6)
    fig.suptitle(f'Source {frame.source_frame}: gray=before, blue=Surface FPS, orange=priority; same per-object axes\n'
                 'Viewport uses observed .5/99.5 percentiles only for display; metrics and sampling retain every valid point. No GT points in these clouds.')
    fig.savefig(output/f'f{frame.source_frame:04d}_clouds.png',dpi=140);plt.close(fig)


def quantitative(output,summary,conditions):
    def get(method,n=256,r=.25,t=20,s=0):
        return summary[next(k for k,v in conditions.items() if v==dict(method=method,num_points=n,structural_ratio=0. if method=='S0' else r,orientation_threshold_deg=t,seed=s))]
    fig,axes=plt.subplots(2,3,figsize=(17,9),constrained_layout=True)
    for method in ('S0','S1','S2','S3','S4'):
        for ax,name in zip(axes[0,:2],('peg','box')):
            values=[get(method,n=n)['objects'][name]['structural']['2px_3mm']['sampling_retention'] for n in (64,128,256,512)]
            ax.plot([64,128,256,512],values,'o-',label=method)
            ax.set(title=name+' structure retained / tracker-observed',xlabel='Total point budget',ylabel='Joint coverage retention')
        ratios=[0.,.125,.25,.5]
        for name,ls in [('peg','-'),('box','--')]:
            axes[0,2].plot(ratios,[get(method,r=r)['objects'][name]['structural']['2px_3mm']['sampling_retention'] for r in ratios],ls,marker='.',label=method+' '+name)
            axes[1,0].plot(ratios,[get(method,r=r)['objects'][name]['surface']['all']['distance_mm']['p95'] for r in ratios],ls,marker='.',label=method+' '+name)
        row=get(method)
        total_priority=sum(v['counts']['priority'] for v in row['objects'].values())
        fp=sum(v['negative']['known_negative_union']['priority'] for v in row['objects'].values())
        axes[1,1].bar(method,fp/sum(v['counts']['selected'] for v in row['objects'].values()))
        axes[1,2].bar(method,row['timings_ms']['sampling_total']['p95'])
    axes[0,2].set(title='Ratio vs structure retention',xlabel='Structural ratio',ylabel='Retention')
    axes[1,0].set(title='Ratio vs all-observed surface p95',xlabel='Structural ratio',ylabel='Source to sampled distance (mm)')
    axes[1,1].set(title='Known-negative priority slots / all selected',ylabel='Fraction (overlapping ROIs unioned)')
    axes[1,2].set(title='Sampling latency at 256 nodes',ylabel='CPU p95 ms (no GT/I/O/plots)')
    for ax in axes.flat:ax.grid(alpha=.2)
    for ax in axes[0]:ax.legend(fontsize=7,ncol=2)
    axes[1,0].legend(fontsize=7,ncol=2)
    fig.savefig(output/'tradeoffs.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    for method in ('S1','S2','S3','S4'):
        for ax,name in zip(axes[:2],('peg','box')):
            ax.plot([10,20,30,40],[get(method,t=t)['objects'][name]['structural']['2px_3mm']['sampling_retention'] for t in (10,20,30,40)],'o-',label=method)
            ax.set(title=name+' threshold diagnostic',xlabel='Angle threshold (deg)',ylabel='Retention')
        for name,ls in [('peg','-'),('box','--')]:
            vals=[get(method,s=s)['objects'][name]['structural']['2px_3mm']['sampling_retention'] for s in (0,1,2)]
            axes[2].plot([0,1,2],vals,ls,marker='.',label=method+' '+name)
    axes[2].set(title='Three FPS seeds at 256/.25/20',xlabel='FPS seed',ylabel='Retention')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
    fig.savefig(output/'sensitivity.png',dpi=150);plt.close(fig)
