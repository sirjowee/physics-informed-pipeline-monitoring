"""Reference re-implementation of two 2024 IJSRED pipeline-monitoring papers.

Subpackages:
- ``leak_detection``: physics-informed neural network for leak detection in
  multiphase pipelines, with black-box and RTTM baselines (paper IJSRED-V7I6P142).
- ``shm_classification``: 1D-CNN classification of simulated |S11| traces
  (paper IJSRED-V7I6P143).

All data in this package is synthetic. See RESULTS.md for what reproduced.
"""

__version__ = "0.1.0.dev0"
