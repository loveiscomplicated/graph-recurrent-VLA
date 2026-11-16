"""Shared sizing for episode collection jobs."""

DEFAULT_BATCH_EPISODES = 8
DEFAULT_WORKERS = 2


def collection_defaults(config):
    """Resolve and validate collection sizing without mutating the caller's config."""
    settings = dict(batch_episodes=DEFAULT_BATCH_EPISODES, workers=DEFAULT_WORKERS)
    settings.update(config)
    for key in ("batch_episodes", "workers"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    return settings


def iter_batches(items, size):
    """Yield consecutive groups, including a final partial group."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def iter_episode_jobs(job, items, settings, *, episodes_per_job=1):
    """Run ordered batches with one isolated process per active simulator job."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    jobs_per_batch = max(1, settings["batch_episodes"] // episodes_per_job)
    workers = min(settings["workers"], jobs_per_batch, len(items))
    if workers <= 1:
        yield from map(job, items)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        for batch in iter_batches(items, jobs_per_batch):
            yield from pool.map(job, batch)
