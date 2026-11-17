"""Timer contracts with deterministic clocks; no simulator or accelerator needed."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import Mock, patch

from utils.timer import Timer


class TimerTests(unittest.TestCase):
    def test_multiple_functions_and_repeated_names(self):
        timer = Timer()

        @timer.wrap(name="preprocess")
        def preprocess(value, *, offset=0):
            return value + offset

        model = timer.wrap(lambda value: value * 2, name="model")
        with patch("utils.timer.perf_counter", side_effect=[0, 2, 3, 7, 8, 14]):
            self.assertEqual(preprocess(1, offset=2), 3)
            self.assertEqual(model(3), 6)
            self.assertEqual(preprocess(4), 4)

        self.assertEqual(preprocess.__name__, "preprocess")
        self.assertEqual(timer.summary()["preprocess"], {
            "count": 2, "total_seconds": 8, "mean_seconds": 4,
            "min_seconds": 2, "max_seconds": 6,
        })
        self.assertEqual(timer.summary()["model"]["total_seconds"], 4)

    def test_nested_and_recursive_regions_are_inclusive(self):
        timer = Timer()
        with patch("utils.timer.perf_counter", side_effect=[0, 1, 3, 5]):
            with timer.measure("region"):
                with timer.measure("region"):
                    pass
        self.assertEqual(timer.summary()["region"]["count"], 2)
        self.assertEqual(timer.summary()["region"]["total_seconds"], 7)

    def test_callable_instance_and_exception_preservation(self):
        timer = Timer()
        failure = ValueError("original failure")

        class Model:
            def __call__(self, value):
                raise failure

        model = Model()
        wrapped = timer.wrap(model, name="model")
        with self.assertRaises(ValueError) as caught:
            wrapped(1)
        self.assertIs(caught.exception, failure)
        self.assertIs(wrapped.__wrapped__, model)
        self.assertEqual(timer.summary()["model"]["count"], 1)

    def test_synchronization_precedes_start_and_stop(self):
        events = []
        timer = Timer(synchronize=lambda: events.append("sync"))

        def clock():
            events.append("clock")
            return len(events)

        with patch("utils.timer.perf_counter", side_effect=clock):
            with timer.measure("gpu_work"):
                events.append("work")
        self.assertEqual(events, ["sync", "clock", "work", "sync", "clock"])
        self.assertEqual(timer.summary()["gpu_work"]["total_seconds"], 3)

    def test_summary_is_snapshot_and_reset_clears_aggregates(self):
        timer = Timer()
        with timer.measure("work"):
            pass
        snapshot = timer.summary()
        snapshot["work"]["count"] = 99
        self.assertEqual(timer.summary()["work"]["count"], 1)
        timer.reset()
        self.assertEqual(timer.summary(), {})

    def test_concurrent_calls_are_all_recorded(self):
        timer = Timer()
        wrapped = timer.wrap(lambda value: value + 1, name="work")
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(wrapped, range(100)))
        self.assertEqual(results, list(range(1, 101)))
        self.assertEqual(timer.summary()["work"]["count"], 100)

    def test_async_function_includes_awaited_execution(self):
        timer = Timer()

        @timer.wrap(name="async_work")
        async def work():
            await asyncio.sleep(0)
            return 42

        coroutine = work()
        self.assertEqual(timer.summary(), {})
        with patch("utils.timer.perf_counter", side_effect=[1, 4]):
            self.assertEqual(asyncio.run(coroutine), 42)
        self.assertEqual(timer.summary()["async_work"]["total_seconds"], 3)

    def test_invalid_inputs_fail_before_execution(self):
        timer = Timer()
        target = Mock()
        for name in ("", "  ", None, 1):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    timer.wrap(target, name=name)
                with self.assertRaises(ValueError):
                    with timer.measure(name):
                        target()
        target.assert_not_called()
        with self.assertRaises(TypeError):
            timer.wrap(42, name="invalid")
        with self.assertRaises(TypeError):
            Timer(synchronize=42)

        def generate():
            yield 1

        with self.assertRaisesRegex(TypeError, "generator"):
            timer.wrap(generate, name="generator")


if __name__ == "__main__":
    unittest.main()
