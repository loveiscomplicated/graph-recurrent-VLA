"""Episode-level metrics; contract smokes never produce policy success rates."""
import math


OFFICIAL_NAME = ('Stop-the-clock / Zero-simulation-time-inference-delay / '
                 '20Hz simulation-control Closed-loop Evaluation')


def wilson(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    p, denominator = successes/count, 1+z*z/count
    centre = (p+z*z/(2*count))/denominator
    radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0., centre-radius), min(1., centre+radius)]


def summarize(episodes, *, smoke):
    result = dict(official_name=OFFICIAL_NAME, performance_kind='B',
                  status='ROLLOUT_NOT_EXECUTED' if smoke else 'COMPLETE',
                  policy_success_rate=None, confidence_interval_95=None,
                  smoke_excluded_from_performance=smoke, groups={})
    if smoke:
        result['contract_episodes'] = len(episodes)
        return result
    for group in ('frozen_test', 'heldout'):
        rows = [e for e in episodes if e['group'] == group]
        complete = all(e['status'] == 'COMPLETE' for e in rows)
        successes = sum(e.get('success', False) for e in rows)
        result['groups'][group] = dict(planned=len(rows), completed=sum(e['status']=='COMPLETE' for e in rows),
            status='NOT_EXECUTED' if not rows else 'COMPLETE' if complete else 'INCOMPLETE',
            successes=successes, success_rate=successes/len(rows) if rows and complete else None,
            confidence_interval_95=wilson(successes, len(rows)) if rows and complete else None,
            interval_method='Wilson score, two-sided 95%; independent episode seeds assumed')
        if not complete:
            result['status'] = 'INCOMPLETE'
    return result


def prepare_seeds(provenance, config):
    split, dataset = provenance['split'], provenance['dataset']
    by_id = {e['episode_id']: e for e in dataset['episodes']}
    test_ids = split['splits']['test']
    if len(test_ids) != 12:
        raise ValueError('12 frozen test episodes required')
    heldout = config['heldout_seeds']
    if not 30 <= len(heldout) <= 50:
        raise ValueError('freeze 30..50 held-out seeds before evaluation')
    excluded = set(config['smoke_seeds']) | set(config['excluded_seeds'])
    dataset_seeds = {e['seed'] for e in dataset['episodes']}
    if dataset_seeds & (set(config['smoke_seeds']) | {config['time_contract_seed']}):
        raise ValueError('smoke seeds overlap original dataset')
    excluded |= dataset_seeds | {config['time_contract_seed']}
    if len(heldout) != len(set(heldout)) or excluded & set(heldout):
        raise ValueError('held-out seed overlaps dataset/pilot/smoke or duplicate')
    return dict(version='stop_clock_seeds_v1', frozen=True,
                split_sha256=provenance['identity']['split_sha256'],
                checkpoint_sha256=provenance['manifest']['files']['checkpoint']['sha256'],
                groups=dict(frozen_test=[dict(seed=by_id[e]['seed'], episode_id=e,
                    reset_group=split['reset_groups']['test'][i]) for i, e in enumerate(test_ids)],
                    heldout=[dict(seed=s) for s in heldout]),
                smoke_seeds=config['smoke_seeds'], excluded_seeds=sorted(excluded),
                selection='fixed before any policy success observation; report groups separately')


def validate_seeds(manifest, provenance, config):
    if manifest != prepare_seeds(provenance, config):
        raise ValueError('seed manifest must match frozen protocol, checkpoint and split exactly')
    return [dict(row, group=group) for group, rows in manifest['groups'].items() for row in rows]
