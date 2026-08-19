"""Tests for document_pipeline.extract — PDF extraction and Paperless submission."""

from __future__ import annotations

import email
from email.message import EmailMessage, Message

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


@respx.mock
def test_submit_message_pdfs_sends_decoded_filename():
    route = respx.post("http://paperless/api/documents/post_document/").mock(
        return_value=httpx.Response(200)
    )
    msg = email.message_from_string(
        "Content-Type: multipart/mixed; boundary=b\n"
        "\n"
        "--b\n"
        "Content-Type: application/pdf\n"
        'Content-Disposition: attachment; filename="=?utf-8?q?vorl=C3=A4ufig.pdf?="\n'
        "Content-Transfer-Encoding: base64\n"
        "\n"
        "JVBERi0xLjQgc2FtcGxl\n"
        "--b--\n"
    )
    assert extract.submit_message_pdfs(msg, "http://paperless", "tok") is True
    # httpx writes the filename as raw UTF-8 bytes into the multipart header.
    assert 'filename="vorläufig.pdf"'.encode() in route.calls[0].request.content
