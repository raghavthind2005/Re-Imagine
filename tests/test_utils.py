import json
import random

import numpy as np
import pytest

from mindseye.data.schema import Example
from mindseye.utils.cache import DiskCache, cache_key
from mindseye.utils.runlog import append_jsonl, completed_ids
from mindseye.utils.seeding import rng, set_seed

# --- cache keys ---------------------------------------------------------------


def test_cache_key_is_order_independent():
    a = cache_key(model_id="m", prompt_version="v1", params={"t": 0.0, "max_tokens": 8})
    b = cache_key(params={"max_tokens": 8, "t": 0.0}, prompt_version="v1", model_id="m")
    assert a == b


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "other"},
        {"prompt_version": "v2"},
        {"params": {"t": 0.1}},
        {"inputs": [b"different image"]},
    ],
)
def test_cache_key_changes_with_any_part(change):
    base = {"model_id": "m", "prompt_version": "v1", "params": {"t": 0.0}, "inputs": [b"image"]}
    assert cache_key(**base) != cache_key(**{**base, **change})


def test_cache_key_handles_arrays_by_content():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert cache_key(a=x) == cache_key(a=x.copy())
    assert cache_key(a=x) != cache_key(a=x.reshape(4, 3))
    assert cache_key(a=x) != cache_key(a=x.astype(np.float64))


def test_cache_key_distinguishes_int_from_float():
    assert cache_key(t=1) != cache_key(t=1.0)


def test_cache_key_rejects_unknown_types():
    with pytest.raises(TypeError):
        cache_key(a=object())


def test_disk_cache_round_trip(tmp_path):
    cache = DiskCache(tmp_path)
    key = cache_key(model_id="m", inputs="x")
    assert cache.get_json(key) is None
    assert not cache.has(key)
    cache.put_json(key, {"text": "left", "n": 3})
    assert cache.has(key)
    assert cache.get_json(key) == {"text": "left", "n": 3}
    cache.put_bytes(key, b"\x00\x01", suffix=".npy")
    assert cache.get_bytes(key, suffix=".npy") == b"\x00\x01"
    assert not list(tmp_path.rglob("*.tmp"))


# --- seeding ------------------------------------------------------------------


def test_set_seed_repeats_python_and_numpy():
    set_seed(7)
    first = (random.random(), np.random.rand())
    set_seed(7)
    assert (random.random(), np.random.rand()) == first


def test_rng_is_independent_of_global_state():
    a = rng(3).integers(0, 1000, 5)
    np.random.seed(99)
    assert (rng(3).integers(0, 1000, 5) == a).all()


# --- resumable predictions ----------------------------------------------------


def test_completed_ids_ignores_truncated_last_line(tmp_path):
    path = tmp_path / "predictions.jsonl"
    append_jsonl(path, {"id": "a", "pred": "left"})
    append_jsonl(path, {"id": "b", "pred": "right"})
    with open(path, "a") as f:
        f.write('{"id": "c", "pr')
    assert completed_ids(path) == {"a", "b"}
    assert completed_ids(tmp_path / "missing.jsonl") == set()


# --- schema -------------------------------------------------------------------


def test_example_round_trip():
    ex = Example("sat-1", "SAT", "dev", "Which way?", ["left", "right"], "left", ["a.png"])
    assert Example.from_dict(json.loads(json.dumps(ex.to_dict()))) == ex


@pytest.mark.parametrize(
    "kwargs",
    [
        {"choices": ["only one"]},
        {"media": []},
        {"id": ""},
    ],
)
def test_example_rejects_malformed(kwargs):
    base = {
        "id": "x",
        "benchmark": "SAT",
        "split": "dev",
        "question": "q",
        "choices": None,
        "answer": "1.5",
        "media": ["v.mp4"],
    }
    with pytest.raises(ValueError):
        Example(**{**base, **kwargs})
