"""The package logger must survive Prefect's logging config.

Flow runs execute in a subprocess where Prefect applies its own dictConfig with
`root: level: WARNING`. Module loggers are NOTSET and inherit that, so without
the level set in `document_pipeline/__init__.py` every per-document `logger.info`
is dropped before reaching a handler — silently, and only inside flow runs, which
is exactly where those lines are wanted.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture
def prefect_logging_applied():
    """Apply Prefect's real logging config, then restore what was there."""
    import document_pipeline  # noqa: F401 — the import under test
    from prefect.logging.configuration import setup_logging

    root = logging.getLogger()
    saved_level, saved_handlers = root.level, list(root.handlers)
    setup_logging()
    yield
    root.setLevel(saved_level)
    root.handlers[:] = saved_handlers


def test_module_loggers_still_emit_info_under_prefect_logging(prefect_logging_applied, caplog):
    logger = logging.getLogger("document_pipeline.enrich")
    assert logger.getEffectiveLevel() <= logging.INFO, (
        "document_pipeline logger inherits root=WARNING from Prefect's config — "
        "per-document log lines would be dropped inside flow runs"
    )

    with caplog.at_level(logging.INFO, logger="document_pipeline.enrich"):
        logger.info("Document %s retitled -> %r", 42, "Invoice")
    assert "Document 42 retitled" in caplog.text
