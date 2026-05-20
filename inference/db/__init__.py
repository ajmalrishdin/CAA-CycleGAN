"""Unified dataset package for inference.

This module exposes the same loader functions and `DATABASE_REGISTRY`
that the original `db_loaders.py` provided, but split across per-dataset
modules for maintainability.
"""
from .adfecgdb import list_adfecgdb_records, load_adfecgdb_record
from .nifeadb import list_nifeadb_records, load_nifeadb_record
from .nifecg import list_nifecg_records, load_nifecg_record
from .cinc2013 import list_cinc2013_records, load_cinc2013_record
from .ninfea import list_ninfea_records, load_ninfea_record

DATABASE_REGISTRY = {
    'ADFECGDB': {
        'list_fn': list_adfecgdb_records,
        'load_fn': load_adfecgdb_record,
        'has_chest': False,
        'has_annotations': True,
        'description': 'Abdominal & Direct Fetal ECG DB (5 records)',
    },
    'NIFEADB': {
        'list_fn': list_nifeadb_records,
        'load_fn': load_nifeadb_record,
        'has_chest': True,
        'has_annotations': False,
        'description': 'NI-FECG Arrhythmia DB (26 records)',
    },
    'NI-FECG': {
        'list_fn': list_nifecg_records,
        'load_fn': load_nifecg_record,
        'has_chest': True,
        'has_annotations': True,
        'description': 'Non-Invasive Fetal ECG DB (55 records)',
    },
    'CinC2013': {
        'list_fn': list_cinc2013_records,
        'load_fn': load_cinc2013_record,
        'has_chest': False,
        'has_annotations': True,
        'description': 'PhysioNet CinC 2013 Challenge (25 records)',
    },
    'NINFEA': {
        'list_fn': list_ninfea_records,
        'load_fn': load_ninfea_record,
        'has_chest': True,
        'has_annotations': False,
        'description': 'NINFEA Multimodal Foetal ECG (60 records)',
    },
}

__all__ = [
    'list_adfecgdb_records', 'load_adfecgdb_record',
    'list_nifeadb_records', 'load_nifeadb_record',
    'list_nifecg_records', 'load_nifecg_record',
    'list_cinc2013_records', 'load_cinc2013_record',
    'list_ninfea_records', 'load_ninfea_record',
    'DATABASE_REGISTRY',
]
