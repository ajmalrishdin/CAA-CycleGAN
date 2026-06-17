"""Compatibility shim: re-export new `db` package API.

This file preserves the original module import path (`inference/db_loaders.py`)
while delegating implementations to the refactored `inference.db` package.
"""

from db import *

__all__ = [
    'list_adfecgdb_records', 'load_adfecgdb_record',
    'list_nifeadb_records', 'load_nifeadb_record',
    'list_nifecg_records', 'load_nifecg_record',
    'list_cinc2013_records', 'load_cinc2013_record',
    'list_ninfea_records', 'load_ninfea_record',
    'DATABASE_REGISTRY',
]
