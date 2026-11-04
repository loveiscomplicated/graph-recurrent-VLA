"""Change only the surface-edge witness reference in a frozen-input control."""
import ast
import inspect
from pathlib import Path


def with_reference(function):
    """Reuse the installed function AST, replacing exactly two reference reads."""
    original = inspect.unwrap(function)
    tree = ast.parse(inspect.getsource(original))
    definition = tree.body[0]
    definition.args.args.extend([ast.arg(arg='reference_xyz'), ast.arg(arg='reference_objects')])
    replacements = {'xyz[None, None, :, :]': 'reference_xyz[None, None, :, :]',
                    'objects[None, None, :]': 'reference_objects[None, None, :]'}
    counts = {key: 0 for key in replacements}

    class Replace(ast.NodeTransformer):
        def visit_Subscript(self, node):
            key = ast.unparse(node)
            if key in replacements:
                counts[key] += 1
                return ast.copy_location(ast.parse(replacements[key], mode='eval').body, node)
            return self.generic_visit(node)

    tree = Replace().visit(tree)
    assert set(counts.values()) == {1}, counts
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<witness-reference-control>', 'exec'), namespace)
    return namespace[definition.name]


def main(capture, inference, runtime, output):
    import numpy as np
    import torch
    from .common import CAMERAS, PARENTS, SIZES, entries, read, write, sha, sensor_frame
    from .localize_information_loss import Observer
    from graph.plan_a_bc.geometry import CurrentGeometry
    from graph.persistent_geometry.tensor_camera import tensor_camera
    from graph.persistent_geometry.tensor_graph import numpy_graph, FIELDS, surface_edges_tensor
    from graph.persistent_geometry.tensor_records import _identity_ranks
    from graph.persistent_geometry.tensor_fusion import prepare_tensor_views
    from graph.task10i_stage4a2.dinov2_matching_audit import csv_write
    from utils.experiment_logger import ExperimentLogger

    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    control = with_reference(surface_edges_tensor)
    index = entries(capture)
    schema = read(runtime / 'normalizer.json')['schema']
    config = dict(source_sha256=sha(Path(__file__)), model_forwards=0, GT_arrays_used=False,
                  inference_receipt_sha256=sha(inference / 'inference_complete.json'),
                  change='Only witness reference: final nodes -> current pre-FPS dedup pool; '
                         'same final nodes, kNN, normals, thresholds, and depth free-space rule')
    rows = []
    with ExperimentLogger(team='integration', model='task10i_edge_reference_control', seed=42,
                          job_type='evaluate', root=output / 'experiments', config=config) as log:
        for parent in PARENTS:
            for size in SIZES:
                geo = CurrentGeometry(runtime / 'inputs', schema)
                try:
                    for t in range(21):
                        host, data = sensor_frame(capture, index, parent, t, size, f'{parent}:native{size}')
                        cameras = [tensor_camera(c) for c in host]
                        surfaces = geo.fk.surfaces(data['robot76'])
                        data.update(episode=np.array(cameras[0].episode), timestep=np.int64(t),
                                    timestamp=np.float64(t / 20), qpos=data['robot76'][:9],
                                    robot_link_world=np.stack([s.world_from_link for s in surfaces]))
                        with np.load(inference / parent / str(size) / f't{t:04d}.npz') as z:
                            expected = {k: z[k].copy() for k in z.files}
                        masks = {c: torch.as_tensor(expected['raw_masks'][ci], device='cuda')
                                 for ci, c in enumerate(CAMERAS)}
                        with Observer(geo) as observer:
                            graph, records, _ = geo.build(cameras, data, surfaces, lambda: masks)
                        saved = numpy_graph(graph)
                        assert all(np.array_equal(getattr(saved, k).numpy(), expected[k]) for k in FIELDS)
                        pool = observer.data['pool']
                        views = prepare_tensor_views(cameras, observer.data['gate_masks'], coordinate_frame='ee')
                        order = torch.argsort(_identity_ranks(records), stable=True)
                        order = order[torch.argsort(records.object_id[order], stable=True)]
                        xyz, normals = records.xyz[order], records.normals[order]
                        valid, objects = records.normal_valid[order], records.object_id[order]
                        for oid in (0, 1):
                            take = objects == oid
                            args = (xyz[take], normals[take], valid[take], objects[take], views)
                            baseline, audit = surface_edges_tensor(*args)
                            same, _ = control(*args, xyz[take], objects[take])
                            assert torch.equal(baseline, same)
                            reference = pool.object_id == oid
                            changed, changed_audit = control(*args, pool.xyz[reference], pool.object_id[reference])
                            original_pairs = set(map(tuple, baseline.T.cpu().tolist()))
                            changed_pairs = set(map(tuple, changed.T.cpu().tolist()))
                            assert original_pairs <= changed_pairs
                            def isolated(edges):
                                return int(take.sum()) - len(torch.unique(edges))
                            rows.append(dict(parent=parent, timestep=t, size=size, object_id=oid,
                                nodes=int(take.sum()), reference_points=int(reference.sum()),
                                original_edges=int(baseline.shape[1] // 2),
                                pool_reference_edges=int(changed.shape[1] // 2),
                                original_isolated=isolated(baseline), pool_reference_isolated=isolated(changed),
                                original_witness_reject=int(audit['rejected_witness']),
                                pool_reference_witness_reject=int(changed_audit['rejected_witness']),
                                original_free_reject=int(audit['rejected_free_space']),
                                pool_reference_free_reject=int(changed_audit['rejected_free_space'])))
                    print('EDGE_REFERENCE_CONTROL', parent, size, flush=True)
                finally:
                    geo.close()
        csv_write(output / 'edge_reference_control.csv', rows)
        summary = []
        for size in SIZES:
            for oid in (0, 1):
                group = [r for r in rows if r['size'] == size and r['object_id'] == oid]
                keys = [k for k in group[0] if k not in ('parent', 'timestep', 'size', 'object_id')]
                summary.append(dict(size=size, object_id=oid, frames=len(group),
                                    means={k: float(np.mean([r[k] for r in group])) for k in keys}))
        write(output / 'summary.json', dict(status='COMPLETE', run_id=log.run_id, config=config,
              summary=summary, graph_frames=168, object_controls=len(rows),
              all_five_graph_tensors_exact=True, all_same_reference_edges_exact=True,
              all_original_edges_retained=True, edge_GT_accuracy_claimed=False))
        log.log(dict(graph_frames=168, object_controls=len(rows), model_forwards=0))
        log.link_output(output / 'summary.json')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    for name in ('capture', 'inference', 'runtime', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    main(args.capture, args.inference, args.runtime, args.output)
