"""Runs one function over many items on a thread pool, logging progress."""
from concurrent.futures import ThreadPoolExecutor

from pycolmap import logging


def map_with_progress(function, items, description, num_threads, log_every):
    """Yields `function(item)` for each of `items`, in order, computed on
    `num_threads` worker threads. The caller stays on its own thread, so it can
    own resources the workers must not touch, such as the COLMAP database."""
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        for count, result in enumerate(pool.map(function, items), start=1):
            yield result
            if count % log_every == 0:
                logging.info(f"{description} {count}/{len(items)}")
