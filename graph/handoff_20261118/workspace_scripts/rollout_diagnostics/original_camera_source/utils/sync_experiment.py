"""Upload local JSON events when no usable W&B offline bundle was created.

Run once per experiment after restoring connectivity and credentials. Existing
W&B runs are rejected to avoid silently duplicating metric history.
"""
import argparse
import json
import os
from pathlib import Path

from .experiment_logger import has_wandb_api_key, load_wandb_env, project_name


def sync(directory):
    load_wandb_env()
    import wandb

    directory = Path(directory).resolve()
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata['status'] == 'running':
        raise ValueError('finish the experiment before synchronizing')
    entity = os.environ.get('WANDB_ENTITY') or metadata.get('entity')
    if not has_wandb_api_key() or not entity:
        raise ValueError('W&B credentials and WANDB_ENTITY are required; set WANDB_API_KEY through dotenv or the environment')
    config = json.loads((directory / 'config.yaml').read_text())
    events = json.loads((directory / 'metrics.json').read_text())
    run = wandb.init(project=metadata.get('project') or project_name(config), entity=entity, id=metadata['run_id'],
                     group=metadata['group'], job_type=metadata['job_type'],
                     config=dict(parameters=config, experiment=metadata), mode='online',
                     resume='never', dir=str(directory),
                     settings=wandb.Settings(init_timeout=10, save_code=False,
                                             disable_git=True, console='off'))
    failed = True
    try:
        for event in events:
            values = dict(event['metrics'])
            if event['step'] is not None: values['epoch'] = event['step']
            run.log(values)
        failed = metadata['status'] == 'failed'
    finally:
        run.finish(exit_code=1 if failed else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    sync(parser.parse_args().directory)


if __name__ == '__main__': main()
