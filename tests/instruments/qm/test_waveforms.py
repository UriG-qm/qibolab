"""Tests ``config/pulses.py``."""

import warnings

import pytest

pytest.importorskip("qm")

from qibolab._core.instruments.qm.config.pulses import waveforms_from_pulse
from qibolab._core.pulses import Pulse
from qibolab._core.pulses.envelope import Rectangular


def test_sub_minimum_duration_emits_userwarning():
    """Sub-minimum-duration pulses should emit a UserWarning."""
    pulse = Pulse(duration=8, amplitude=0.1, envelope=Rectangular())
    with pytest.warns(UserWarning, match="below the QM minimum"):
        waveforms_from_pulse(pulse, sampling_rate=1, max_voltage=0.5)


def test_over_minimum_duration_does_not_warn():
    """Over-minimum-duration pulses should not emit a warning."""
    pulse = Pulse(duration=40, amplitude=0.1, envelope=Rectangular())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Should not raise — no UserWarning for valid duration
        waveforms_from_pulse(pulse, sampling_rate=1, max_voltage=0.5)
