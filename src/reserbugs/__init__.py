"""
reserBUGS
Python Library to execute predictive models of pollinators abundance
"""

from .reservoir_computing import ReservoirComputing
from .data import CopernicusDataRetriever, ModisDataRetriever, ModisRetrieverConfig
from .visualization import visualizations
from .evaluation import type_s_error, type_m_error, scoring_rules

__all__ = [
    "ReservoirComputing",
    "CopernicusDataRetriever",
    "ModisDataRetriever",
    "ModisRetrieverConfig",
    "visualizations",
    "type_s_error",
    "type_m_error",
    "scoring_rules",
]

__version__ = "0.1.0"