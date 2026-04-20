from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qm import qua
from qm.qua import Cast, declare, declare_stream, fixed, wait
from qm.qua._dsl import _ResultSource, _Variable  # for type declaration only

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import ChannelId


def _collect(i, q, npulses):
    """Collect I and Q components of signal.

    I and Q should be the the last dimension of the returned array,
    except when multiple results are acquired to the same stream in the
    instrument, when they should be second to last.
    """
    signal = np.stack([i, q])
    return np.moveaxis(signal, 0, -1 - int(npulses > 1))


def _split(data, npulses):
    """Split results of different readout pulses to list.

    These results were acquired in the same acquisition stream in the
    instrument.
    """
    if npulses == 1:
        return [data]
    return list(np.moveaxis(data, -1, 0))


@dataclass
class Acquisition(ABC):
    """QUA variables used for saving of acquisition results.

    This class can be instantiated only within a QUA program scope. Each
    readout pulse is associated with its own set of acquisition
    variables.
    """

    operation: str
    element: str
    """Element from QM ``config`` that the pulse will be applied on."""
    average: bool
    keys: list[int] = field(default_factory=list)
    stream: Optional[str] = None
    """User-chosen stream base name.

    When non-None, stream variables are named ``<stream>_I`` /
    ``<stream>_Q`` instead of the channel-derived ``self.name``.
    When None (default) the original channel-collapsed behaviour is
    preserved.
    """
    save: bool = True
    """When False, skip ``qua.save`` and ``stream_processing`` entries.

    The ``measure`` instruction is still emitted.  Default is True,
    preserving current behaviour.
    """

    @property
    def name(self):
        """Identifier to download results from the instruments."""
        # FIXME: QUA 1.2.1a2 and OPX1000 don't like `/` character in stream processing ``save``
        return f"{self.operation}_{self.element}".replace("/", "|")

    @property
    def stream_name(self) -> str:
        """Effective stream base name used for QUA stream allocation.

        Returns the user-supplied :attr:`stream` when set, otherwise
        falls back to the channel-derived :attr:`name`.  This is the
        single override point: all stream label construction in subclasses
        should use ``self.stream_name`` rather than ``self.name`` directly.
        """
        return self.stream if self.stream is not None else self.name

    @property
    def npulses(self):
        return len(self.keys)

    @abstractmethod
    def declare(self):
        """Declares QUA variables related to this acquisition.

        Assigns acquisition variables to the corresponding QM
        controller. This was proposed by QM to avoid crashes.
        """

    @abstractmethod
    def measure(self, operation):
        """Send measurement pulse and acquire results.

        Args:
            operation (str): Operation (from ``config``) corresponding to the pulse to be played.
        """

    @abstractmethod
    def download(self, *dimensions):
        """Save streams to prepare for fetching from host device.

        Args:
            dimensions (int): Dimensions to use for buffer of data.
        """

    @abstractmethod
    def fetch(self):
        """Fetch downloaded streams to host device."""


def raw_to_volts(signal):
    """Convert raw acquisition to volts."""
    return signal / 4096


def assign_variables_to_element(element, *variables):
    """Forces the given variables to be used by the given element thread.

    Useful as a workaround for when the compiler wrongly assigns
    variables which can cause gaps.

    Taken from qualang_tools library.
    """
    _exp = Cast.to_int(variables[0])
    for variable in variables[1:]:
        _exp += Cast.to_int(variable)
    wait(4 + 0 * _exp, element)


@dataclass
class RawAcquisition(Acquisition):
    """QUA variables used for raw waveform acquisition."""

    adc_stream: Optional[_ResultSource] = None
    """Stream to collect raw ADC data."""

    def declare(self):
        self.adc_stream = declare_stream(adc_trace=True)

    def measure(self, operation):
        qua.reset_phase(self.element)
        qua.measure(operation, self.element, self.adc_stream)

    def download(self, *dimensions):
        istream = self.adc_stream.input1()
        qstream = self.adc_stream.input2()
        if self.average:
            istream = istream.average()
            qstream = qstream.average()
        istream.save(f"{self.name}_I")
        qstream.save(f"{self.name}_Q")

    def fetch(self, handles):
        ires = handles.get(f"{self.name}_I").fetch_all()
        qres = handles.get(f"{self.name}_Q").fetch_all()
        # convert raw ADC signal to volts
        signal = _collect(raw_to_volts(ires), raw_to_volts(qres), self.npulses)
        return _split(signal, self.npulses)


@dataclass
class IntegratedAcquisition(Acquisition):
    """QUA variables used for integrated acquisition."""

    i: Optional[_Variable] = None
    q: Optional[_Variable] = None
    """Variables to save the (I, Q) values acquired from a single shot."""
    istream: Optional[_ResultSource] = None
    qstream: Optional[_ResultSource] = None
    """Streams to collect the results of all shots."""

    def declare(self):
        self.i = declare(fixed)
        self.q = declare(fixed)
        self.istream = declare_stream()
        self.qstream = declare_stream()
        assign_variables_to_element(self.element, self.i, self.q)

    def measure(self, operation):
        qua.measure(
            operation,
            self.element,
            None,
            qua.dual_demod.full("cos", "out1", "sin", "out2", self.i),
            qua.dual_demod.full("minus_sin", "out1", "cos", "out2", self.q),
        )
        if self.save:
            qua.save(self.i, self.istream)
            qua.save(self.q, self.qstream)

    def download(self, *dimensions):
        if not self.save:
            return
        istream = self.istream
        qstream = self.qstream
        if self.npulses > 1:
            istream = istream.buffer(self.npulses)
            qstream = qstream.buffer(self.npulses)
        for dim in dimensions:
            istream = istream.buffer(dim)
            qstream = qstream.buffer(dim)
        if self.average:
            istream = istream.average()
            qstream = qstream.average()
        istream.save(f"{self.stream_name}_I")
        qstream.save(f"{self.stream_name}_Q")

    def fetch(self, handles):
        ires = handles.get(f"{self.stream_name}_I").fetch_all()
        qres = handles.get(f"{self.stream_name}_Q").fetch_all()
        signal = _collect(ires, qres, self.npulses)
        return _split(signal, self.npulses)


@dataclass
class ShotsAcquisition(Acquisition):
    """QUA variables used for shot classification.

    Threshold and angle must be given in order to classify shots.
    """

    threshold: Optional[float] = None
    """Threshold to be used for classification of single shots."""
    angle: Optional[float] = None
    """Angle in the IQ plane to be used for classification of single shots."""

    i: Optional[_Variable] = None
    q: Optional[_Variable] = None
    """Variables to save the (I, Q) values acquired from a single shot."""
    shot: Optional[_Variable] = None
    """Variable for calculating an individual shots."""
    shots: Optional[_ResultSource] = None
    """Stream to collect multiple shots."""

    def __post_init__(self):
        self.cos = np.cos(self.angle)
        self.sin = np.sin(self.angle)

    def declare(self):
        self.i = declare(fixed)
        self.q = declare(fixed)
        self.shot = declare(int)
        self.shots = declare_stream()
        assign_variables_to_element(self.element, self.i, self.q, self.shot)

    def measure(self, operation):
        qua.measure(
            operation,
            self.element,
            None,
            qua.dual_demod.full("cos", "out1", "sin", "out2", self.i),
            qua.dual_demod.full("minus_sin", "out1", "cos", "out2", self.q),
        )
        qua.assign(
            self.shot,
            qua.Cast.to_int(self.i * self.cos - self.q * self.sin > self.threshold),
        )
        if self.save:
            qua.save(self.shot, self.shots)

    def download(self, *dimensions):
        if not self.save:
            return
        shots = self.shots
        if self.npulses > 1:
            shots = shots.buffer(self.npulses)
        for dim in dimensions:
            shots = shots.buffer(dim)
        if self.average:
            shots = shots.average()
        shots.save(f"{self.stream_name}_shots")

    def fetch(self, handles):
        shots = handles.get(f"{self.stream_name}_shots").fetch_all()
        return _split(shots, self.npulses)


ACQUISITION_TYPES = {
    AcquisitionType.RAW: RawAcquisition,
    AcquisitionType.INTEGRATION: IntegratedAcquisition,
    AcquisitionType.DISCRIMINATION: ShotsAcquisition,
}


def create_acquisition(
    operation: str,
    element: str,
    options: ExecutionParameters,
    threshold: float,
    angle: float,
    stream: Optional[str] = None,
    save: bool = True,
) -> Acquisition:
    """Create container for the variables used for saving acquisition in the
    QUA program.

    Args:
        stream: Optional user-chosen stream base name.  When non-None,
            stream variables are named ``<stream>_I`` / ``<stream>_Q``
            instead of the channel-derived default.
        save: When False, ``qua.save`` and ``stream_processing`` entries
            are skipped; the ``measure`` instruction is still emitted.

    Returns:
        ``Acquisition`` object containing acquisition variables.
    """
    average = options.averaging_mode is AveragingMode.CYCLIC
    kwargs: dict = {"stream": stream, "save": save}
    if options.acquisition_type is AcquisitionType.DISCRIMINATION:
        kwargs.update({"threshold": threshold, "angle": angle})
    acquisition = ACQUISITION_TYPES[options.acquisition_type](
        operation, element, average, **kwargs
    )
    return acquisition


Acquisitions = dict[tuple[str, ChannelId], Acquisition]
