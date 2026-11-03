"""Synthetic |S11| data for the SHM classification paper."""

from .s11_simulator import (
    CLASS_NAMES,
    CLASS_SIGNATURES,
    N_POINTS,
    TraceParams,
    add_gaussian_noise,
    frequency_grid,
    generate_dataset,
    s11_trace_db,
    sample_trace_params,
    stratified_split,
    zscore_per_trace,
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_SIGNATURES",
    "N_POINTS",
    "TraceParams",
    "add_gaussian_noise",
    "frequency_grid",
    "generate_dataset",
    "s11_trace_db",
    "sample_trace_params",
    "stratified_split",
    "zscore_per_trace",
]
