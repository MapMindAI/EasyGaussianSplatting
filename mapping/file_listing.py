"""File listings that stay fast on network mounts."""

import os
from pathlib import Path


def files_under(directory):
    """Every file below `directory`, from directory listings alone.

    Checking files one at a time costs a round trip each on a mount such as
    sshfs; listing a directory costs one for all of its entries.
    """
    return [
        Path(root) / name
        for root, _, names in os.walk(directory, followlinks=True)
        for name in names
    ]
