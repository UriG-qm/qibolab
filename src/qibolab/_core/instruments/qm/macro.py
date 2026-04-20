"""QM-scoped escape hatch: emit raw QUA imperatives in a PulseSequence slot.

``QuaMacro`` is an abstract pydantic Model that users subclass to inject
arbitrary ``qm.qua`` imperatives into qibolab's QM emission loop. It
lives under the QM instrument tree because QUA is QM-only; no
cross-backend portability is intended or implied.

``QuaEmissionContext`` is a runtime dataclass threaded through the QM
emission loop and passed to ``QuaMacro.emit``. It exposes already-
declared sweep QUA variables, helpers for declaring new ones and
allocating streams, element-name resolution, and a re-entry callback
so block-wrapper macros can delegate body emission back to qibolab.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from qm.qua import declare, declare_stream
from qm.qua.type_hints import QuaVariable

from qibolab._core.serialize import Model

if TYPE_CHECKING:
    from qibolab._core.sequence import PulseSequence


class QuaMacro(Model):
    """Escape hatch: emit raw QUA inside a PulseSequence slot.

    Subclasses implement ``emit(ctx)`` and use ``ctx`` to access already-
    declared QUA variables, declare new ones, allocate streams, and call
    arbitrary ``qm.qua`` imperatives. Lives under the QM instrument tree
    because QUA is QM-only; no cross-backend portability is intended.

    No ``model_config`` redeclaration — inherits from ``Model`` (which
    already sets ``frozen=True`` / ``extra="forbid"``). Subclasses that
    genuinely need arbitrary-type fields must opt in explicitly via
    ``model_config = ConfigDict(**Model.model_config, arbitrary_types_allowed=True)``
    to avoid silently dropping inherited flags.
    """

    @abstractmethod
    def emit(self, ctx: QuaEmissionContext) -> None:
        """Emit QUA imperatives for this macro.

        Called by the QM backend's emission loop when a
        ``QuaMacroInstruction`` wrapping this macro is encountered.
        """
        ...


@dataclass
class QuaEmissionContext:
    """Runtime handle passed to ``QuaMacro.emit``.

    Gives read access to sweep / acquisition QUA vars, write helpers for
    declaring new vars and streams, and element-name resolution. Does not
    re-export ``qm.qua`` imperatives — macros import them directly.
    """

    # Handles to already-declared QUA vars, keyed by a stable logical name
    # the user chose in their Python code. Two families live in one dict:
    #
    # * Sweep vars: keyed by the sweeper's logical name — i.e. the value of
    #   ``Sweeper.name`` when set (from qm-qa/25). Sweepers without a name
    #   are not registered; macros that need them must surface a clear error
    #   ("set name= on the Sweeper to read it from a macro").
    # * Acquisition result vars: keyed as ``<acquisition-stream-name>_I`` /
    #   ``<acquisition-stream-name>_Q`` — matching (a)#19's single-underscore
    #   I/Q convention on stream-per-readout names.
    qua_vars: dict[str, QuaVariable] = field(default_factory=dict)

    # All element names the program knows about (drive / flux /
    # acquisition / probe), resolved from the sequence's channel ids
    # via qibolab's existing channel->QUA-element mapping.
    elements: frozenset[str] = field(default_factory=frozenset)

    # Record of streams this context has allocated, keyed by the
    # user-provided logical name. Values are the ``declare_stream()``
    # handles. Macros may read this to reuse streams across call sites.
    streams: dict[str, Any] = field(default_factory=dict)

    # Back-reference so macros that wrap a child PulseSequence (e.g.
    # strict_timing_) can re-enter the qibolab emission loop. Set by the
    # emission loop at construction; not a user-facing parameter — macros
    # access it only via ``ctx.emit_sequence(inner)``.
    _emit_sequence: Callable[[PulseSequence], None] | None = None

    def declare(
        self,
        type_: type,
        *,
        size: int | None = None,
        value: Any = None,
        name: str | None = None,
    ) -> QuaVariable:
        """Declare a new QUA variable; register it in ``qua_vars`` if
        ``name`` is given so later macros can look it up.

        The return type is ``QuaVariable`` (from ``qm.qua.type_hints``),
        which qua 1.2.5 parameterises by numeric type (e.g.
        ``QuaVariable[int]``, ``QuaVariable[fixed]``, ``QuaVariable[bool]``).
        Subclasses may narrow the signature where the element type is
        statically known.

        Raises ``ValueError`` if ``name`` is given and a variable with
        that name has already been declared via this context — to reuse an
        existing named var, read ``ctx.qua_vars[name]`` directly.
        """
        var = declare(type_, size=size, value=value)
        if name is not None:
            if name in self.qua_vars:
                raise ValueError(
                    f"QUA variable {name!r} already declared; pick another "
                    "name or reuse the existing handle via ctx.qua_vars."
                )
            self.qua_vars[name] = var
        return var

    def declare_stream(self, name: str, **kwargs: Any) -> Any:
        """Allocate a new QUA stream; register it in ``streams``.

        ``name`` follows the macro-stream convention ``<macro>__<logical>``
        (double underscore) — e.g. ``ComputeRamseyPhase__diag``,
        ``ActiveReset__shot_count``. The double-underscore separator keeps
        macro-allocated streams unambiguously distinct from (a)#19's
        acquisition streams, which use single-underscore ``_I`` / ``_Q``
        suffixes on readout-stream names.

        Raises if ``name`` does not contain ``__`` or collides with an
        existing stream.
        """
        if "__" not in name:
            raise ValueError(
                "macro stream names must contain '__' separator "
                "(convention: <macro>__<logical>)"
            )
        if name in self.streams:
            raise ValueError(f"stream {name!r} already allocated")
        stream = declare_stream(**kwargs)
        self.streams[name] = stream
        return stream

    def emit_sequence(self, inner: PulseSequence) -> None:
        """Re-enter the qibolab emission loop for a child sequence.

        Used by block-wrapper macros (e.g. ``StrictTimingMacro``) so they
        do not have to re-implement pulse emission manually. The macro
        calls this inside any QUA block it wants (``with strict_timing_():``,
        ``with if_(...)``, etc.).
        """
        if self._emit_sequence is None:
            raise RuntimeError(
                "emit_sequence callback is not set; "
                "QuaEmissionContext was not constructed by the QM emission loop."
            )
        self._emit_sequence(inner)
