"""Typed error hierarchy for DP-HONEY.

Every failure mode raises a subclass of :class:`DPHoneyError`. This lets the CLI
map a single base type to a clean nonzero exit while callers that care can still
distinguish specific conditions (bad privacy parameters vs. artifact drift vs.
schema corruption). We never raise bare ``Exception`` and never swallow errors
silently.
"""

from __future__ import annotations


class DPHoneyError(Exception):
    """Base class for every error raised by DP-HONEY."""


class UnknownFormatError(DPHoneyError):
    """A requested format slug is not present in the registry."""


class InvalidPrivacyParameter(DPHoneyError):
    """A privacy/training parameter (``epsilon`` or ``clip``) is invalid."""


class EmptyCorpusError(DPHoneyError):
    """Training was attempted with an empty corpus."""


class FormatRepairError(DPHoneyError):
    """Bounded repair could not produce a spec-valid token within the limit."""


class CountLimitError(DPHoneyError):
    """A requested batch count is outside the supported range."""


class ModelArtifactDecodeError(DPHoneyError):
    """A model artifact is not well-formed JSON."""


class ModelSchemaError(DPHoneyError):
    """A model artifact is structurally invalid (fields, alphabet, or probabilities)."""


class FormatSpecMismatchError(DPHoneyError):
    """An artifact's embedded format snapshot/hash has drifted from the registry."""


class ModelArtifactExistsError(DPHoneyError):
    """Saving would overwrite an existing artifact and ``force`` was not set."""
