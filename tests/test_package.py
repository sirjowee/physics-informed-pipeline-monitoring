"""Smoke test: the package imports and exposes a version."""

import pipeline_monitoring


def test_version_string():
    assert isinstance(pipeline_monitoring.__version__, str)
    assert pipeline_monitoring.__version__
