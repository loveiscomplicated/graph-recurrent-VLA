import unittest

from .profile import Profiler, instrument_graph_edges


def surface_edges_tensor(value):
    return value*2, {'checked': True}


def toy_graph(value):
    nodes = value+1
    intra, audit = surface_edges_tensor(nodes)
    edges = intra+3
    attrs = edges*2
    metadata = audit
    return nodes, edges, attrs, metadata


class Tests(unittest.TestCase):
    def test_nested_spans_are_exclusive_and_natural_API_is_kept(self):
        p = Profiler(lambda:None, detailed=False)
        with p.span('ignored', 'off'):
            with p.span('API', 'scan', cpu=True, always=True):
                with p.span('API', 'request', cpu=True, always=True):
                    pass
        self.assertEqual(len(p.events), 2)
        parent = p.events[-1]
        self.assertAlmostEqual(sum(e['exclusive_ms'] for e in p.events), parent['wall_ms'])

    def test_edge_timer_preserves_function_results_and_boundary(self):
        p = Profiler(lambda:None, detailed=True)
        measured = instrument_graph_edges(toy_graph, p.span)
        for value in (-1, 0, 17):
            self.assertEqual(measured(value), toy_graph(value))
        self.assertEqual([e['stage'] for e in p.events], ['17']*3)


if __name__ == '__main__':
    unittest.main()
