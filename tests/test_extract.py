"""Tests for document_pipeline.extract — PDF extraction and Paperless submission."""

from __future__ import annotations

from email.message import EmailMessage

import httpx
import respx

from document_pipeline import extract


def _pdf_message() -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = "sender@example.com"
    msg["Subject"] = "Invoice"
    msg.set_content("See attached.")
    msg.add_attachment(
        b"%PDF-1.4 sample",
        maintype="application",
        subtype="pdf",
        filename="invoice.pdf",
    )
    return msg


@respx.mock
def test_submit_message_pdfs_submits_pdf_and_returns_true():
    route = respx.post("http://paperless/api/documents/post_document/").mock(
        return_value=httpx.Response(200)
    )
    result = extract.submit_message_pdfs(_pdf_message(), "http://paperless", "tok")
    assert result is True
    assert route.called
    body = route.calls[0].request.content
    assert b"invoice.pdf" in body
    assert b"%PDF-1.4 sample" in body


def test_submit_message_pdfs_returns_false_when_no_pdf():
    msg = EmailMessage()
    msg.set_content("just text, no attachments")
    assert extract.submit_message_pdfs(msg, "http://paperless", "tok") is False


@respx.mock
def test_submit_message_pdfs_sends_auth_token():
    route = respx.post("http://paperless/api/documents/post_document/").mock(
        return_value=httpx.Response(200)
    )
    extract.submit_message_pdfs(_pdf_message(), "http://paperless", "mytoken")
    assert route.calls[0].request.headers["authorization"] == "Token mytoken"
