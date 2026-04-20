"""Canary tests for the QuaMacro extension point (qm-qa/24)."""

import pytest

pytest.importorskip("qm")

from unittest.mock import MagicMock

from qibolab._core.instruments.qm.macro import QuaEmissionContext, QuaMacro
from qibolab._core.pulses.pulse import QuaMacroInstruction
from qibolab._core.sequence import PulseSequence


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class NoOpMacro(QuaMacro):
    """Trivial QuaMacro subclass that does nothing."""

    def emit(self, ctx: QuaEmissionContext) -> None:
        pass


class SpyMacro(QuaMacro):
    """QuaMacro subclass that records emit() invocations."""

    call_count: int = 0
    last_ctx: object = None

    model_config = {"frozen": False, "extra": "forbid", "arbitrary_types_allowed": True}

    def emit(self, ctx: QuaEmissionContext) -> None:
        object.__setattr__(self, "call_count", self.call_count + 1)
        object.__setattr__(self, "last_ctx", ctx)


# ---------------------------------------------------------------------------
# Test 1: QuaMacro can be subclassed
# ---------------------------------------------------------------------------


def test_quamacro_subclass():
    """QuaMacro can be subclassed and instantiated."""
    macro = NoOpMacro()
    assert isinstance(macro, QuaMacro)


# ---------------------------------------------------------------------------
# Test 2: QuaMacroInstruction can be instantiated
# ---------------------------------------------------------------------------


def test_quamacro_instruction_instantiation():
    """QuaMacroInstruction(macro=NoOpMacro()) can be created."""
    macro = NoOpMacro()
    instr = QuaMacroInstruction(macro=macro)
    assert instr.kind == "qua_macro"
    assert instr.duration == 0.0
    assert instr.macro is macro


# ---------------------------------------------------------------------------
# Test 3: QuaMacroInstruction pydantic round-trip
# ---------------------------------------------------------------------------


def test_quamacro_instruction_roundtrip():
    """QuaMacroInstruction round-trips through model_dump / model_validate."""
    instr = QuaMacroInstruction(macro=NoOpMacro())
    dumped = instr.model_dump()
    assert dumped["kind"] == "qua_macro"
    # macro is Any — model_dump serialises it via NoOpMacro's own model_dump
    assert "macro" in dumped

    # Re-validate: model_validate expects the macro field to have a 'kind'-like
    # discriminator for the Any type — since macro is typed Any, pydantic will
    # accept the dict or object. We verify the round-trip at the instruction level.
    validated = QuaMacroInstruction.model_validate(dumped)
    assert validated.kind == "qua_macro"


# ---------------------------------------------------------------------------
# Test 4: PulseSequence containing QuaMacroInstruction round-trips
# ---------------------------------------------------------------------------


def test_pulse_sequence_with_macro_roundtrip():
    """A PulseSequence containing a QuaMacroInstruction round-trips via pydantic."""
    from pydantic import TypeAdapter

    instr = QuaMacroInstruction(macro=NoOpMacro())
    seq = PulseSequence([("q0/drive", instr)])

    # PulseSequence is a UserList with a custom pydantic schema; use TypeAdapter.
    adapter = TypeAdapter(PulseSequence)
    dumped = adapter.dump_python(seq)
    validated = adapter.validate_python(dumped)

    assert len(validated) == 1
    ch, pulse = validated[0]
    assert str(ch) == "q0/drive"
    assert pulse.kind == "qua_macro"


# ---------------------------------------------------------------------------
# Test 5: Spy macro records emit(ctx) invocation with a fake context
# ---------------------------------------------------------------------------


def test_spy_macro_emit_called():
    """A spy QuaMacro records that emit(ctx) was called with the context."""
    spy = SpyMacro()
    # Build a fake QuaEmissionContext without entering a real program() block.
    ctx = QuaEmissionContext(
        qua_vars={"tau": MagicMock()},
        elements=frozenset(["q0/drive"]),
        streams={},
        _emit_sequence=None,
    )
    assert spy.call_count == 0
    spy.emit(ctx)
    assert spy.call_count == 1
    assert spy.last_ctx is ctx
