from collections import OrderedDict

import numpy as np


def make_dict_cached_function(func):
    cache = {}
    stats = {"requests": 0, "computed": 0}

    def evaluate(x):
        stats["requests"] += 1
        if x not in cache:
            cache[x] = func(x)
            stats["computed"] += 1
        return cache[x]

    return evaluate, cache, stats


def make_hashmap_cached_function(func, maxsize=256):
    cache = OrderedDict()
    stats = {"requests": 0, "computed": 0, "evicted": 0}

    def evaluate(x):
        stats["requests"] += 1

        if x in cache:
            cache.move_to_end(x)
            return cache[x]

        value = func(x)
        cache[x] = value
        stats["computed"] += 1

        if maxsize is not None and len(cache) > maxsize:
            cache.popitem(last=False)
            stats["evicted"] += 1

        return value

    return evaluate, cache, stats


def make_cached_nd_function(func):
    cache = {}
    stats = {"requests": 0, "computed": 0}

    def make_key(x):
        array = np.asarray(x, dtype=float)
        if array.ndim == 0:
            return ("scalar", float(array))
        return ("vector", tuple(float(value) for value in array.ravel()))

    def evaluate(x):
        key = make_key(x)
        stats["requests"] += 1
        if key not in cache:
            value = float(func(np.asarray(x, dtype=float)))
            cache[key] = value
            stats["computed"] += 1
        return cache[key]

    return evaluate, cache, stats
