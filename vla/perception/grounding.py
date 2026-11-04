"""Terra-only is the default; legacy auth fallback remains explicit."""
from vla.perception.live_grounding import LiveGrounding, policy_identity


def _auth_fallback(config):
    value = (config or {}).get('grounding', {})
    if not isinstance(value, dict) or set(value) - {'policy'}:
        raise ValueError('grounding config supports only an explicit policy')
    policy = value.get('policy', 'terra_only')
    if policy not in ('terra_only', 'terra_auth_fallback'):
        raise ValueError('unknown grounding policy')
    return policy == 'terra_auth_fallback'


def grounding_identity(config=None):
    return policy_identity(auth_fallback=_auth_fallback(config))


def grounding_from_config(config=None, **kwargs):
    if 'auth_fallback' in kwargs:
        raise ValueError('choose grounding.policy explicitly')
    return LiveGrounding(**kwargs, auth_fallback=_auth_fallback(config))
