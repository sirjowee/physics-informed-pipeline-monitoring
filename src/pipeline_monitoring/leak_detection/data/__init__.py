"""Synthetic two-fluid pipeline data for the leak-detection paper."""

from .config import (
    REGIME_NAMES,
    REGIMES,
    DatasetConfig,
    PipelineConfig,
    RegimeParams,
    ScenarioSpec,
    SimulationConfig,
)
from .scenarios import (
    SENSOR_NAMES,
    add_sensor_noise,
    event_split,
    generate_dataset,
    load_dataset,
    sample_scenarios,
    save_dataset,
)
from .two_fluid_solver import SimulationOutput, mass_residual_steady, primitives, simulate

__all__ = [
    "REGIMES",
    "REGIME_NAMES",
    "SENSOR_NAMES",
    "DatasetConfig",
    "PipelineConfig",
    "RegimeParams",
    "ScenarioSpec",
    "SimulationConfig",
    "SimulationOutput",
    "add_sensor_noise",
    "event_split",
    "generate_dataset",
    "load_dataset",
    "mass_residual_steady",
    "primitives",
    "sample_scenarios",
    "save_dataset",
    "simulate",
]
