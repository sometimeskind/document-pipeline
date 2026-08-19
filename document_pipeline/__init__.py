"""document-pipeline: Prefect-driven IMAP/Paperless integration."""

import logging

# Flow runs execute in a subprocess where Prefect applies its own logging config
# (prefect/logging/logging.yml), which sets the ROOT logger to WARNING. Module
# loggers here are NOTSET, so they inherit that and every `logger.info(...)` in
# scan.py, extract.py, webdav.py and enrich.py is dropped before it reaches a
# handler — the per-document detail lines never appeared in a flow run's logs.
#
# `logging.basicConfig` in cli.py cannot fix this: it only runs in the service
# process, never in the run subprocess.
#
# Setting the level on the package logger is enough to get these records to the
# console handler on root. Pair it with PREFECT_LOGGING_EXTRA_LOGGERS=document_pipeline
# in the deployment to also attach Prefect's APILogHandler, which is what puts
# them in the flow run's log in the Prefect UI.
logging.getLogger(__name__).setLevel(logging.INFO)
