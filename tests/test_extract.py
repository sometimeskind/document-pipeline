"""Tests for document_pipeline.extract — PDF extraction into the WebDAV scan queue."""

from __future__ import annotations

import email
from email.message import EmailMessage, Message
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from document_pipeline import extract
from document_pipeline.webdav import WebDAVClient


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


@pytest.fixture
def queue():
    return MagicMock(spec=WebDAVClient)


def test_queue_message_pdfs_puts_each_pdf_under_the_uid(queue):
    count = extract.queue_message_pdfs(_pdf_message(), "42", queue, "/homes/scanner/mail")

    assert count == 1
    queue.put.assert_called_once_with("/homes/scanner/mail/42-invoice.pdf", b"%PDF-1.4 sample")


def test_queue_message_pdfs_returns_zero_when_no_pdf(queue):
    msg = EmailMessage()
    msg.set_content("just text, no attachments")

    assert extract.queue_message_pdfs(msg, "42", queue, "/homes/scanner/mail") == 0
    queue.put.assert_not_called()


def test_queue_message_pdfs_propagates_a_failed_put(queue):
    """The caller decides what an unqueued message means (it stays unflagged);
    swallowing the error here would flag a message whose PDF never landed."""
    queue.put.side_effect = httpx.ConnectError("stalwart down")

    with pytest.raises(httpx.ConnectError):
        extract.queue_message_pdfs(_pdf_message(), "42", queue, "/homes/scanner/mail")


def test_queue_message_pdfs_keeps_two_same_named_attachments_apart(queue):
    msg = _pdf_message()
    msg.add_attachment(b"%PDF-1.4 second", maintype="application", subtype="pdf", filename="invoice.pdf")

    assert extract.queue_message_pdfs(msg, "42", queue, "/homes/scanner/mail") == 2
    assert [c.args[0] for c in queue.put.call_args_list] == [
        "/homes/scanner/mail/42-invoice.pdf", "/homes/scanner/mail/42-invoice-2.pdf"
    ]


def test_queue_message_pdfs_never_posts_to_paperless(queue):
    with respx.mock(assert_all_mocked=True) as router:
        extract.queue_message_pdfs(_pdf_message(), "42", queue, "/homes/scanner/mail")
        assert not router.calls


def _raw_pdf_part(disposition_params: str) -> Message:
    """A PDF part parsed the way production parses it (compat32, not EmailMessage).

    imap_client uses email.message_from_bytes, so get_filename() sees exactly
    what the sender wrote — which is the whole point of #1297.
    """
    return email.message_from_string(
        "Content-Type: application/pdf\n"
        f"Content-Disposition: attachment; {disposition_params}\n"
        "Content-Transfer-Encoding: base64\n"
        "\n"
        "JVBERi0xLjQgc2FtcGxl\n"
    )


def test_attachment_filename_decodes_q_encoded_word():
    part = _raw_pdf_part(
        'filename="=?utf-8?q?2026=5F07=5F31=5Fvorl=C3=A4ufige=5FRechnung.pdf?="'
    )
    assert extract._attachment_filename(part) == "2026_07_31_vorläufige_Rechnung.pdf"


def test_attachment_filename_decodes_b_encoded_word():
    part = _raw_pdf_part('filename="=?utf-8?b?dsO2Z2VsLnBkZg==?="')
    assert extract._attachment_filename(part) == "vögel.pdf"


def test_attachment_filename_keeps_rfc2231_continuation_working():
    part = _raw_pdf_part(
        "filename*0*=utf-8''vorl%C3%A4ufige; filename*1*=_Rechnung.pdf"
    )
    assert extract._attachment_filename(part) == "vorläufige_Rechnung.pdf"


def test_attachment_filename_leaves_plain_ascii_alone():
    part = _raw_pdf_part('filename="invoice.pdf"')
    assert extract._attachment_filename(part) == "invoice.pdf"


def test_attachment_filename_falls_back_when_absent():
    part = _raw_pdf_part("")
    assert extract._attachment_filename(part) == "attachment.pdf"


def test_attachment_filename_falls_back_on_malformed_encoded_word():
    # Unknown charset raises LookupError, bad base64 raises HeaderParseError.
    # Neither may cost the document, so the raw value is used instead.
    for raw in ("=?bogus-charset?q?abc.pdf?=", "=?utf-8?b?!!!notbase64!!!?="):
        part = _raw_pdf_part(f'filename="{raw}"')
        # Ugly, but the document still reaches paperless with a .pdf suffix.
        assert extract._attachment_filename(part) == f"{raw}.pdf"


def test_attachment_filename_strips_rfc2231_charset_remnant():
    # A sender wrapped a whole `charset''value` in an encoded word.
    part = _raw_pdf_part("filename=\"=?utf-8?q?utf-8''Leistungs=C3=BCbersicht.pdf?=\"")
    assert extract._attachment_filename(part) == "Leistungsübersicht.pdf"


def test_attachment_filename_strips_path_separators_and_control_chars():
    part = _raw_pdf_part('filename="=?utf-8?q?..=2F..=2Fetc=2Fpasswd=5Cx=00y.pdf?="')
    # Separators become underscores, the NUL is dropped, and the leading dots
    # go with it — nothing here can escape the directory paperless writes to.
    assert extract._attachment_filename(part) == "_.._etc_passwd_xy.pdf"


def test_attachment_filename_caps_length_but_keeps_suffix():
    part = _raw_pdf_part(f'filename="{"a" * 400}.pdf"')
    result = extract._attachment_filename(part)
    assert result == "a" * 120 + ".pdf"


def test_attachment_filename_appends_pdf_when_missing():
    part = _raw_pdf_part('filename="=?utf-8?q?Leistungs=C3=BCbersicht?="')
    assert extract._attachment_filename(part) == "Leistungsübersicht.pdf"


def test_queue_message_pdfs_uses_the_decoded_filename_with_a_pdf_suffix(queue):
    msg = email.message_from_string(
        "Content-Type: multipart/mixed; boundary=b\n"
        "\n"
        "--b\n"
        "Content-Type: application/pdf\n"
        'Content-Disposition: attachment; filename="=?utf-8?q?vorl=C3=A4ufig?="\n'
        "Content-Transfer-Encoding: base64\n"
        "\n"
        "JVBERi0xLjQgc2FtcGxl\n"
        "--b--\n"
    )
    assert extract.queue_message_pdfs(msg, "7", queue, "/q") == 1
    # `.pdf` is what keeps the object inside the scan flow's eligibility allowlist.
    queue.put.assert_called_once_with("/q/7-vorläufig.pdf", b"%PDF-1.4 sample")
