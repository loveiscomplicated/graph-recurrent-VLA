"""Reproducible live collision/graph audit before any learned comparison."""
import argparse
from pathlib import Path
from time import perf_counter
import resource
import statistics
import torch
from ..common import seed_all, write_json, versions, config
from ..data import load_dataset, sha256
from ..graph_builder import GraphBuilder, EdgeType
from ..maniskill_gt import validate_live_collision
from ..models import parameter_count
from ..v2.models import CommonPolicy
from ..v2.diagnostics import ReplaySession
from .interaction_graph import InteractionGraph, Relation, policy
from .geometry_audit import regions, REGIONS, boundary_check, resample
from .edge_diagnostics import diagnose
from .visualization import draw, histogram, aperture_view

STATES = [('initial', 0), ('acquired', 56), ('transport', 72), ('alignment', 77)]


def timing(fn, repeats=20):
    for _ in range(3): fn()
    samples = []
    for _ in range(repeats):
        start = perf_counter(); fn(); samples.append((perf_counter()-start)*1000)
    return dict(median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples), repeats=repeats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('outputs/graph/v3'))
    p.add_argument('--figures', action='store_true')
    p.add_argument('--peg-points', type=int, default=64); p.add_argument('--hole-points', type=int, default=64)
    p.add_argument('--sampling-weights', type=float, nargs=4, default=(1, 3, 3, 1))
    p.add_argument('--entry-band', type=float, default=.02)
    args = p.parse_args(); seed_all(0, 4)
    data, dataset_geometry = load_dataset('outputs/graph/dataset.pt')
    ep = next(e for e in data['manifest']['episodes'] if e['episode_id'] == 29)
    out = args.output; (out/'audit').mkdir(parents=True, exist_ok=True); (out/'visualizations').mkdir(exist_ok=True)
    builders = {'original': GraphBuilder(), 'interaction': InteractionGraph(), 'sampling': InteractionGraph()}
    report = dict(versions=versions(), dataset_sha256=sha256('outputs/graph/dataset.pt'), backend=None,
                  torch_device='cpu', configuration=dict(radius_m=.05, local_k=8, cross_k=4,
                  peg_points=args.peg_points, hole_points=args.hole_points, sampling_weights=args.sampling_weights, entry_band=args.entry_band), states={})
    session = ReplaySession(data)
    from ..maniskill_gt import GTAdapter
    session.adapter = GTAdapter(args.peg_points, args.hole_points, 17)
    graphs = {v:{} for v in builders}
    try:
        report['backend'] = repr(session.env.unwrapped.backend)
        for stage, prefix in STATES:
            error, fingerprint = session.prepare(ep, prefix); g = session.adapter(session.env)
            record = dict(prefix=prefix, prefix_observation_error=error, fingerprint=fingerprint,
                          live_collision=validate_live_collision(session.env, g), variants={})
            report['states'][stage] = record
            for variant, builder in builders.items():
                state = resample(g, args.hole_points, args.sampling_weights, entry_band=args.entry_band) if variant == 'sampling' else g
                gr = builder(state); graphs[variant][stage] = gr
                labels = regions(state)
                record['variants'][variant] = dict(graph=diagnose(gr, [e.name for e in (EdgeType if variant=='original' else Relation)]),
                                                  boundary=boundary_check(state), regions={name:int((labels == i).sum()) for i,name in enumerate(REGIONS)})
                hole_relative = state.hole_local-state.hole_offset[:, None]
                near_entry = (labels[:, 3+state.peg_local.shape[1]:] == 4) & \
                             ((hole_relative[..., 1:].abs().amax(-1)-state.hole_radius).abs() <= .02)
                record['variants'][variant]['entry_points_within_2cm_aperture'] = int(near_entry.sum())
                torch.save(dict(geometry=state.as_dict(), edge_index=gr.edge_index, edge_attr=gr.edge_attr, edge_type=gr.edge_type), out/'audit'/f'{stage}_{variant}.pt')
                if args.figures:
                    for mode, selected, goals, zoom in [('geometry', [], False, False), ('local', [0,1], False, False),
                                                       ('interaction', [2,3], False, True), ('global', [4] if variant!='original' else [], True, False)]:
                        draw(state, gr, out/'visualizations'/f'{stage}_{variant}_{mode}.png', f'ep29 prefix {prefix} / {variant} / {mode}', selected, zoom, goals)
            write_json(out/'audit/results.json', report)
            print(f'audited {stage}: '+str({v:record['variants'][v]['graph']['edges_total'] for v in builders}), flush=True)
        # Three additional randomized live geometries, independent of test metrics.
        randomization = []
        for seed in (0, 1, 2):
            session.env.reset(seed=seed, options={'reconfigure':True})
            state = session.adapter(session.env)
            randomization.append(dict(seed=seed, half_size=state.peg_half_size.tolist(),
                                      live_collision=validate_live_collision(session.env, state), boundary=boundary_check(state)))
        report['randomization'] = randomization
    finally: session.close()
    if args.figures:
        for variant in builders: histogram(graphs[variant], out/'visualizations'/f'{variant}_edge_lengths.png')
        from ..geometry import GeometryState
        state = GeometryState(**torch.load(out/'audit/initial_original.pt', weights_only=True)['geometry'])
        sampled = GeometryState(**torch.load(out/'audit/initial_sampling.pt', weights_only=True)['geometry'])
        aperture_view(state, sampled, out/'visualizations/hole_aperture.png')
        from ..visualization import plt
        fig, axes = plt.subplots(2,2,figsize=(11,7))
        for ax,stage in zip(axes.flat, dict(STATES)):
            for variant in builders:
                degrees = report['states'][stage]['variants'][variant]['graph']['incoming_degree'][0]
                ax.hist(degrees,bins=range(0,85,3),histtype='step',label=variant)
            ax.set(title=stage,xlabel='incoming degree',ylabel='node count');ax.legend()
        fig.tight_layout();fig.savefig(out/'visualizations/degree_histogram.png',dpi=140);plt.close(fig)
    # Poses from uniformly spaced dataset rows; graph tensors never contain actions.
    extra = torch.linspace(0, len(data['actions'])-1, 24).long()
    report['additional_pose_checks'] = {}
    for variant, builder in builders.items():
        state = dataset_geometry.select(extra)
        if variant == 'sampling': state = resample(state, entry_band=args.entry_band)
        diag = diagnose(builder(state), [e.name for e in (EdgeType if variant=='original' else Relation)])
        report['additional_pose_checks'][variant] = dict(components=diag['components'], duplicates=diag['duplicates'], self_loops=diag['self_loops'], boundary=boundary_check(state))
    report['episode29_dataset_timeline'] = []
    rows = torch.where(data['episode_id'] == 29)[0]
    for row in rows[::5]:
        gr = builders['interaction'](dataset_geometry.select(row.reshape(1)))
        report['episode29_dataset_timeline'].append(dict(timestep=int(data['timestep'][row]),
            finger_peg=int((gr.edge_type==2).sum()),peg_hole=int((gr.edge_type==3).sum()),global_goal=int((gr.edge_type==4).sum())))
    cfg = config(Path(__file__).parents[1]/'v2/config.toml')
    report['benchmarks'] = {}
    for variant in ('original', 'interaction', 'sampling'):
        model = CommonPolicy('surface', cfg) if variant=='original' else policy(cfg)
        model.eval(); builder = builders[variant]
        results = dict(parameters=parameter_count(model), batches={})
        for batch in (1,32):
            state = dataset_geometry.select(torch.arange(batch))
            if variant=='sampling': state = resample(state, entry_band=args.entry_band)
            gr = builder(state)
            with torch.inference_mode():
                inference = timing(lambda:model(gr, torch.zeros(batch,31)))
                construction = timing(lambda:builder(state))
            size = sum(t.numel()*t.element_size() for t in (gr.x,gr.context,gr.edge_index,gr.edge_attr,gr.edge_type,gr.node_type))
            results['batches'][batch] = dict(construction=construction,inference=inference,graph_tensor_bytes=size,
                                            edges=int(gr.edge_index.shape[1]),nodes_per_graph=int(gr.x.shape[1]))
        model.train(); gr=builder(dataset_geometry.select(torch.arange(2)))
        loss=model(gr,torch.zeros(2,31)).square().mean();loss.backward()
        results['all_parameter_gradients_finite'] = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
        report['benchmarks'][variant]=results
    # macOS ru_maxrss is bytes; process peak includes loaded data, plots and models.
    import sys
    report['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    preserved = out/'audit/preserved_sha256.json'
    if preserved.exists():
        report['preservation']={path:sha256(path)==digest for path,digest in __import__('json').loads(preserved.read_text()).items()}
    write_json(out/'audit/results.json',report)


if __name__=='__main__': main()
