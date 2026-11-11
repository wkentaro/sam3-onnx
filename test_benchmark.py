import collections
import time
import types

from benchmark import Timings
from benchmark import _measure
from benchmark import _measure_warm
from benchmark import _run_with_iobinding
from benchmark import _timed


def test_measure_buffered_frames() -> None:
    timings: Timings = collections.defaultdict(list)
    model = _timed(timings, "model", lambda: time.sleep(0.01), lambda: None)

    def predictor():
        # As the video predictors do: hold the first frames back, then yield
        # them together without running a model.
        buffered = []
        for frame in range(4):
            model()
            buffered.append(frame)
        yield from buffered

    report = _measure(predictor, timings, lambda: None)
    # 10 ms per frame, although the frames arrive with no time between them.
    assert 10 <= report["model"] < 20
    assert report["model"] <= report["total"] < 20


def test_measure_includes_preparation() -> None:
    timings: Timings = collections.defaultdict(list)

    def make_steps():
        # As upstream's init_state loads every frame before the first one is
        # tracked, and returns a generator that has not started.
        time.sleep(0.04)
        return iter(range(4))

    report = _measure(make_steps, timings, lambda: None)
    assert 10 <= report["total"] < 20


def test_timed_excludes_nested_calls() -> None:
    timings: Timings = collections.defaultdict(list)
    inner = _timed(timings, "inner", lambda: time.sleep(0.02), lambda: None)
    outer = _timed(timings, "outer", lambda: (time.sleep(0.01), inner()), lambda: None)
    outer()
    assert 20 <= timings["inner"][0] < 30
    assert 10 <= timings["outer"][0] < 20


def test_measure_warm_times_a_fresh_second_run() -> None:
    timings: Timings = collections.defaultdict(list)
    runs: list[int] = []

    def make_steps():
        # The first run of a shape compiles, which the warm-up absorbs.
        runs.append(len(runs))
        seconds = 0.05 if runs[-1] == 0 else 0.01
        model = _timed(timings, "model", lambda: time.sleep(seconds), lambda: None)
        for _ in range(2):
            model()
            yield

    report = _measure_warm(make_steps, timings, lambda: None)
    assert runs == [0, 1]
    assert 10 <= report["model"] < 20


def test_run_with_iobinding_reads_smallest_output_on_webgpu() -> None:
    read: list[tuple[int, ...]] = []

    def output(*shape: int):
        return types.SimpleNamespace(
            shape=lambda: list(shape), numpy=lambda: read.append(shape)
        )

    binding = types.SimpleNamespace(get_outputs=lambda: [output(1, 256), output(1, 4)])
    for provider, expected in [("WebGpuExecutionProvider", [(1, 4)]), ("x", [])]:
        read.clear()
        session = types.SimpleNamespace(
            get_providers=lambda: [provider], run_with_iobinding=lambda b: None
        )
        _run_with_iobinding(session)(binding)
        assert read == expected


if __name__ == "__main__":
    test_measure_buffered_frames()
    test_measure_includes_preparation()
    test_measure_warm_times_a_fresh_second_run()
    test_timed_excludes_nested_calls()
    test_run_with_iobinding_reads_smallest_output_on_webgpu()
    print("ok")
