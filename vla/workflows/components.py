"""Explicit component selection shared by cache, training and live evaluation."""
from dataclasses import dataclass
from importlib import import_module
from typing import Callable


def _resolve(path):
    if not isinstance(path, str) or path.count(':') != 1:
        raise ValueError('component factory must be a module:attribute string')
    module, name = path.split(':')
    if not module or not name or any(not part.isidentifier() for part in module.split('.') + name.split('.')):
        raise ValueError('invalid component factory path')
    value = import_module(module)
    for part in name.split('.'):
        value = getattr(value, part)
    if not callable(value):
        raise ValueError('component factory is not callable')
    return value


def _name(factory):
    if factory is None:
        return None
    return factory.__module__ + ':' + factory.__qualname__


@dataclass(frozen=True)
class LiveComponents:
    """Factories receive config, never a model output or future observation.

    Encoder factory: factory(model_config). Controller: factory(input_dim,
    model_config). Graph builder: the current (assets, schema, coordinate_frame)
    constructor. Config factory strings are trusted project code, not a sandbox.
    """
    graph_builder: Callable | None = None
    graph_encoder: Callable | None = None
    controller: Callable | None = None
    graph_feature_dim: int | None = None
    presence_dim: int = 2
    state_batch_dim: int | None = None

    def geometry(self, assets, schema, *, coordinate_frame='ee'):
        from graph.current import CurrentGeometry
        factory = CurrentGeometry if self.graph_builder is None else self.graph_builder
        result = factory(assets, schema, coordinate_frame=coordinate_frame)
        required = ('build', 'from_h5', 'reset', 'close')
        if any(not callable(getattr(result, name, None)) for name in required):
            raise TypeError('graph builder must implement build/from_h5/reset/close')
        return result

    def model(self, schema, variant, model_config):
        from vla.integration.live_policy import CachedGraphPolicy
        from torch import nn
        config = dict(model_config)
        encoder = None if self.graph_encoder is None else self.graph_encoder(dict(config))
        if self.graph_encoder is not None and not isinstance(encoder, nn.Module):
            raise TypeError('graph encoder factory must return nn.Module')
        feature_dim = self.graph_feature_dim
        if feature_dim is None:
            feature_dim = getattr(encoder, 'output_dim', 5*config['d'])
        input_dim = feature_dim + self.presence_dim + schema['dimension']
        controller = (None if self.controller is None else self.controller(input_dim, dict(config)))
        if self.controller is not None and not isinstance(controller, nn.Module):
            raise TypeError('controller factory must return nn.Module')
        return CachedGraphPolicy(schema, variant, **config, graph_encoder=encoder,
            controller=controller, graph_feature_dim=feature_dim, presence_dim=self.presence_dim,
            state_batch_dim=self.state_batch_dim)

    def identity(self, *, graph_only=False):
        result = {'graph_builder': _name(self.graph_builder) or 'graph.current:CurrentGeometry'}
        if not graph_only:
            result.update(graph_encoder=_name(self.graph_encoder) or 'gnn.edge_mlp:GraphEncoder/live21',
                controller=_name(self.controller) or 'controller.recurrent:RNNPolicy/gru',
                graph_feature_dim=self.graph_feature_dim, presence_dim=self.presence_dim,
                state_batch_dim=self.state_batch_dim)
        return result

    def validate_checkpoint(self, checkpoint):
        actual = checkpoint.get('components')
        expected = self.identity()
        if actual is None:
            if expected != LiveComponents().identity():
                raise ValueError('checkpoint lacks identity for non-baseline components')
        elif actual != expected:
            raise ValueError('checkpoint component selection changed')


def components_from_config(config):
    values = dict(config.get('components', {}))
    allowed = {'graph_builder', 'graph_encoder', 'controller',
               'graph_feature_dim', 'presence_dim', 'state_batch_dim'}
    if set(values) - allowed:
        raise ValueError('unknown component setting: ' + ', '.join(sorted(set(values)-allowed)))
    for key in ('graph_builder', 'graph_encoder', 'controller'):
        if values.get(key) is not None:
            values[key] = _resolve(values[key])
    return LiveComponents(**values)
