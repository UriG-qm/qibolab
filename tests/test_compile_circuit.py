"""Tests for QibolabBackend.compile_circuit() method.

Verifies that compile_circuit does not touch hardware and produces
valid pulse sequences without platform connection.
"""

import pytest
from qibo import gates
from qibo.models import Circuit

pytest.importorskip("qibolab")

from qibolab._core.backends import QibolabBackend
from qibolab._core.sequence import PulseSequence


def test_compile_circuit_basic():
    """Test that compile_circuit returns a valid PulseSequence without hardware."""
    backend = QibolabBackend("dummy")
    circuit = Circuit(2)
    circuit.add(gates.GPI2(0, phi=0))
    circuit.add(gates.GPI2(1, phi=0))
    circuit.add(gates.M(0, 1))

    sequence, measurement_map = backend.compile_circuit(circuit)

    assert isinstance(sequence, PulseSequence)
    assert isinstance(measurement_map, dict)
    assert len(measurement_map) > 0


def test_compile_circuit_single_qubit():
    """Test compile_circuit with a single-qubit circuit."""
    backend = QibolabBackend("dummy")
    circuit = Circuit(1)
    circuit.add(gates.GPI(0, phi=0.5))
    circuit.add(gates.M(0))

    sequence, measurement_map = backend.compile_circuit(circuit)

    assert isinstance(sequence, PulseSequence)
    assert isinstance(measurement_map, dict)


def test_compile_circuit_no_platform_connect():
    """Verify compile_circuit doesn't call platform.connect()."""
    backend = QibolabBackend("dummy")
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, phi=0))
    circuit.add(gates.M(0))

    # Track if connect() gets called by wrapping it
    connect_called = False
    original_connect = backend.platform.connect

    def mock_connect():
        nonlocal connect_called
        connect_called = True
        original_connect()

    backend.platform.connect = mock_connect

    # compile_circuit should not trigger a connection
    sequence, measurement_map = backend.compile_circuit(circuit)

    assert not connect_called, "compile_circuit should not call platform.connect()"


def test_compile_circuit_invalid_input():
    """Test that compile_circuit rejects non-Circuit inputs."""
    backend = QibolabBackend("dummy")

    with pytest.raises(TypeError):
        backend.compile_circuit("not a circuit")

    with pytest.raises(TypeError):
        backend.compile_circuit([1, 2, 3])

    with pytest.raises(TypeError):
        backend.compile_circuit(None)
