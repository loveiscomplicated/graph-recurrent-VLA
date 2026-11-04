"""Measurement-only spans; no production implementation or policy changes."""
import ast
from contextlib import contextmanager
from functools import wraps
import inspect
import threading
import time


class Profiler:
    def __init__(self, synchronize, *, detailed):
        self.synchronize = synchronize
        self.detailed = detailed
        self.local = threading.local()
        self.events = []
        self.frame = None

    @contextmanager
    def span(self, stage, label, *, cpu=False, always=False):
        if not self.detailed and not always:
            yield
            return
        stack = getattr(self.local, 'stack', None)
        if stack is None:
            stack = self.local.stack = []
        if not cpu:
            self.synchronize()
        event = dict(stage=stage, label=label, frame=self.frame,
                     thread=threading.current_thread().name, children_ns=0)
        stack.append(event)
        event['start_ns'] = time.perf_counter_ns()
        try:
            yield
        finally:
            if not cpu:
                self.synchronize()
            event['end_ns'] = time.perf_counter_ns()
            event['wall_ms'] = (event['end_ns']-event['start_ns'])/1e6
            event['exclusive_ms'] = (event['end_ns']-event['start_ns']-event['children_ns'])/1e6
            assert stack.pop() is event
            if stack:
                stack[-1]['children_ns'] += event['end_ns']-event['start_ns']
            self.events.append(event)

    def wrap(self, function, stage, label, *, cpu=False, always=False):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with self.span(stage, label, cpu=cpu, always=always):
                return function(*args, **kwargs)
        return wrapped


def instrument_graph_edges(function, span):
    """Wrap the existing edge block in a timer, preserving its exact AST.

    No copied edge implementation: statements from the current function's
    intra/audit assignment through the final attrs assignment are reused.
    Fail rather than silently measuring a changed/unrecognized boundary.
    """
    module = ast.parse(inspect.getsource(function))
    definition, = module.body
    assert isinstance(definition, ast.FunctionDef)
    body = definition.body
    begin = [i for i, node in enumerate(body) if isinstance(node, ast.Assign)
             and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Name)
             and node.value.func.id == 'surface_edges_tensor']
    end = [i for i, node in enumerate(body) if isinstance(node, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == 'attrs' for t in node.targets)]
    assert len(begin) == len(end) == 1 and begin[0] < end[0]
    block = body[begin[0]:end[0]+1]
    context = ast.Call(func=ast.Name(id='_stage_timing_span', ctx=ast.Load()),
                       args=[ast.Constant('17'), ast.Constant('all_edges_and_Edge4')], keywords=[])
    definition.body = body[:begin[0]] + [ast.With(
        items=[ast.withitem(context_expr=context)], body=block)] + body[end[0]+1:]
    ast.fix_missing_locations(module)
    # Keep the original module namespace for imported functions and decorators.
    namespace = dict(inspect.unwrap(function).__globals__)
    namespace['_stage_timing_span'] = span
    exec(compile(module, inspect.getsourcefile(function), 'exec'), namespace)
    return namespace[definition.name]
