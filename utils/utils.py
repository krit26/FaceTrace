# Standard Imports
import time
import json
import asyncio
import logging

# Third Party Imports
import yaml


# Internal Imports


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            msg = "%r  %2.2f ms" % (method.__name__, (te - ts) * 1000)
            logging.warning(msg)
        return result

    async def async_timed(*args, **kw):
        ts = time.time()
        result = await method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            msg = "%r  %2.2f ms" % (method.__name__, (te - ts) * 1000)
            logging.warning(msg)
        return result

    if asyncio.iscoroutinefunction(method):
        return async_timed
    return timed


def load_yaml(path):
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data


def load_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def dump_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file)
