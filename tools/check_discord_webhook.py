"""Send one real Discord webhook test message without requiring tmux.

Run from the repository root: python -m tools.check_discord_webhook
"""

import argparse
import os
import sys

from tools.discord_notifier import DEFAULT_ENV_FILE, DiscordTransport
from utils.experiment_logger import load_env_file


def main(argv=None):
    """Load the notifier's credentials, send once, and report the result safely."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    load_env_file(DEFAULT_ENV_FILE)
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL', '').strip()
    if not webhook_url:
        print(
            'DISCORD_WEBHOOK_URL is missing. Set it in the environment or in the '
            'repository-root .env (requires python-dotenv).',
            file=sys.stderr,
        )
        return 2

    try:
        transport = DiscordTransport(webhook_url)
    except ValueError:
        print(
            'DISCORD_WEBHOOK_URL must be a Discord incoming webhook HTTPS URL.',
            file=sys.stderr,
        )
        return 2

    result = transport.post('✅ Discord webhook test: graph-recurrent-VLA notification delivery works.')
    if result.ok:
        print('Test message sent successfully. Check your Discord channel.')
        return 0

    # The shared transport supplies a safe reason without the URL or response body.
    print(f'Test message failed ({result.reason}).', file=sys.stderr)
    if result.delay:
        print(f'Wait at least {result.delay:g} seconds before retrying.', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
