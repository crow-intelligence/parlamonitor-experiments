"""Analysis of Hungarian parliamentary transcripts, cycle 43.

The workspace reads the JSONL exports in ``data/raw/`` and measures them with
the Crow packages (:mod:`saphes`, :mod:`keyflux`, :mod:`kenon`,
:mod:`lexograph`). Nothing here regenerates the exports -- that is the
``backend/export_nlp_datasets.py`` script in the parlamonitor application, and
``data/raw/`` is treated as read-only provenance.
"""

from parlamonitor.loading import (
    DATA_RAW,
    iter_jsonl,
    load_manifest,
    load_qa,
    load_speeches,
    provenance,
)

__all__ = [
    "DATA_RAW",
    "iter_jsonl",
    "load_manifest",
    "load_qa",
    "load_speeches",
    "provenance",
]
