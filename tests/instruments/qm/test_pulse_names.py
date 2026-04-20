"""Tests for optional Pulse.name / operation() user-readable QM operation keys."""

import pytest

pytest.importorskip("qm")

from qibolab._core.pulses import Pulse
from qibolab._core.pulses.envelope import Rectangular
from qibolab._core.instruments.qm.config.pulses import operation


def test_operation_uses_name_when_set():
    p = Pulse(duration=40, amplitude=1.0, envelope=Rectangular(), name="my_pulse")
    assert operation(p) == "my_pulse"


def test_operation_falls_back_to_hash_when_name_none():
    p = Pulse(duration=40, amplitude=1.0, envelope=Rectangular())
    result = operation(p)
    # Fallback format: hash as string (not the name)
    assert result.lstrip("-").isdigit()


def test_pulse_equality_ignores_name():
    p1 = Pulse(duration=40, amplitude=1.0, envelope=Rectangular(), name="a")
    p2 = Pulse(duration=40, amplitude=1.0, envelope=Rectangular(), name="b")
    assert p1 == p2  # same structure, different user-name


def test_pulse_hash_ignores_name():
    p1 = Pulse(duration=40, amplitude=1.0, envelope=Rectangular(), name="a")
    p2 = Pulse(duration=40, amplitude=1.0, envelope=Rectangular(), name="b")
    assert hash(p1) == hash(p2)


def test_unnamed_pulse_equals_named_pulse():
    p_named = Pulse(duration=40, amplitude=1.0, envelope=Rectangular(), name="x")
    p_unnamed = Pulse(duration=40, amplitude=1.0, envelope=Rectangular())
    assert p_named == p_unnamed


def test_operation_named_differs_from_hash():
    """Named and unnamed operations produce different strings."""
    p_named = Pulse(duration=40, amplitude=1.0, envelope=Rectangular(), name="readout")
    p_unnamed = Pulse(duration=40, amplitude=1.0, envelope=Rectangular())
    assert operation(p_named) == "readout"
    assert operation(p_unnamed) != "readout"
