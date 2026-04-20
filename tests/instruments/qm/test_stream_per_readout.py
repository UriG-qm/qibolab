"""Canary tests for opt-in Readout.stream / Readout.save fields.

These tests verify the model-level fields and the Acquisition helpers
without requiring a live QM connection.  They exercise both the default
(backward-compat) path and the explicit opt-in path.
"""

import pytest

pytest.importorskip("qm")

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.instruments.qm.program.acquisition import (
    IntegratedAcquisition,
    create_acquisition,
)
from qibolab._core.pulses import Readout
from qibolab._core.pulses.envelope import Gaussian

# ---------------------------------------------------------------------------
# Readout model field tests (no QM required)
# ---------------------------------------------------------------------------


def test_readout_stream_default_none():
    """Readout.stream defaults to None — backward compat."""
    from qibolab._core.pulses import Acquisition, Pulse

    probe = Pulse(duration=100, amplitude=0.5, envelope=Gaussian(rel_sigma=0.2))
    ro = Readout(
        acquisition=Acquisition(duration=100),
        probe=probe,
    )
    assert ro.stream is None


def test_readout_save_default_true():
    """Readout.save defaults to True — backward compat."""
    from qibolab._core.pulses import Acquisition, Pulse

    probe = Pulse(duration=100, amplitude=0.5, envelope=Gaussian(rel_sigma=0.2))
    ro = Readout(
        acquisition=Acquisition(duration=100),
        probe=probe,
    )
    assert ro.save is True


def test_readout_stream_optin():
    """Readout.stream round-trips through the model."""
    from qibolab._core.pulses import Acquisition, Pulse

    probe = Pulse(duration=100, amplitude=0.5, envelope=Gaussian(rel_sigma=0.2))
    ro = Readout(
        acquisition=Acquisition(duration=100),
        probe=probe,
        stream="readout_g",
    )
    assert ro.stream == "readout_g"


def test_readout_save_false_optin():
    """Readout.save=False round-trips through the model."""
    from qibolab._core.pulses import Acquisition, Pulse

    probe = Pulse(duration=100, amplitude=0.5, envelope=Gaussian(rel_sigma=0.2))
    ro = Readout(
        acquisition=Acquisition(duration=100),
        probe=probe,
        save=False,
    )
    assert ro.save is False


# ---------------------------------------------------------------------------
# Acquisition helper tests
# ---------------------------------------------------------------------------

_OPTIONS_INTEGRATION = ExecutionParameters(
    nshots=10,
    acquisition_type=AcquisitionType.INTEGRATION,
    averaging_mode=AveragingMode.CYCLIC,
)


def test_acquisition_stream_name_default():
    """stream_name falls back to name when stream=None (default path)."""
    acq = create_acquisition("op", "ch", _OPTIONS_INTEGRATION, 0.0, 0.0)
    assert isinstance(acq, IntegratedAcquisition)
    assert acq.stream is None
    assert acq.save is True
    assert acq.stream_name == acq.name


def test_acquisition_stream_name_optin():
    """stream_name uses user-supplied stream when set."""
    acq = create_acquisition(
        "op", "ch", _OPTIONS_INTEGRATION, 0.0, 0.0, stream="readout_g"
    )
    assert isinstance(acq, IntegratedAcquisition)
    assert acq.stream == "readout_g"
    assert acq.stream_name == "readout_g"


def test_acquisition_save_field_forwarded():
    """save=False is forwarded through create_acquisition."""
    acq = create_acquisition("op", "ch", _OPTIONS_INTEGRATION, 0.0, 0.0, save=False)
    assert acq.save is False
