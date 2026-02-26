#!/usr/bin/env python3
"""Minimal standalone LogReader for reading openpilot rlog files.

Replaces the dependency on openpilot's LogReader by bundling the cereal
capnp schemas in nnlc_tools/cereal/ and providing a simple iterator interface.
"""

import bz2
import os

import capnp
import zstandard as zstd

CEREAL_DIR = os.path.join(os.path.dirname(__file__), "cereal")
capnp_log = capnp.load(os.path.join(CEREAL_DIR, "log.capnp"), imports=[CEREAL_DIR])


class LogReader:
    """Read and iterate over messages in an rlog file."""

    def __init__(self, fn, sort_by_time=False):
        with open(fn, "rb") as f:
            dat = f.read()

        # Decompress if needed
        if dat[:4] == b"\x28\xB5\x2F\xFD":  # zstd magic
            dctx = zstd.ZstdDecompressor()
            reader = dctx.stream_reader(dat)
            dat = reader.read()
        elif dat[:2] == b"BZ":
            dat = bz2.decompress(dat)

        self._ents = capnp_log.Event.read_multiple_bytes(dat)

        if sort_by_time:
            self._ents = sorted(self._ents, key=lambda e: e.logMonoTime)

    def __iter__(self):
        return iter(self._ents)
