"""Centralize warning filters for noisy optional graph dependencies."""

from __future__ import annotations

import warnings


_CONFIGURED = False


def configure_warning_filters():
    """Silence known optional-dependency warnings that spam benchmark logs."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    warnings.filterwarnings(
        "ignore",
        message=r"An issue occurred while importing '.*'\. Disabling its usage\.",
        category=UserWarning,
        module=r"torch_geometric(\.|$)",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`torch_geometric\.distributed` has been deprecated since 2\.7\.0.*",
        category=DeprecationWarning,
        module=r"torch_geometric\.llm\.utils\.backend_utils",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The hashes produced for graphs without node or edge attributes changed in v3\.5.*",
        category=UserWarning,
        module=r"networkx\.algorithms\.graph_hashing",
    )

    _CONFIGURED = True
