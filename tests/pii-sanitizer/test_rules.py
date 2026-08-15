from app.rules import apply_rule_based_redaction


def test_ssn_redacted():
    text = "Call the billing line, SSN 123-45-6789 was flagged."
    redacted, manifest = apply_rule_based_redaction(text)
    assert "123-45-6789" not in redacted
    assert "[SSN]" in redacted
    assert any(m["type"] == "SSN" for m in manifest)


def test_phone_variants_redacted():
    for phone in ["(555) 123-4567", "555-123-4567", "+1 555 123 4567", "555.123.4567"]:
        text = f"Call {phone} today"
        redacted, manifest = apply_rule_based_redaction(text)
        assert phone not in redacted
        assert "[PHONE]" in redacted
        assert any(m["type"] == "PHONE" for m in manifest)


def test_email_redacted():
    text = "Reach me at patient.jane@example.com for updates."
    redacted, _manifest = apply_rule_based_redaction(text)
    assert "patient.jane@example.com" not in redacted
    assert "[EMAIL]" in redacted


def test_mrn_redacted():
    text = "Patient MRN: 889910012 was checked in."
    redacted, _manifest = apply_rule_based_redaction(text)
    assert "889910012" not in redacted
    assert "[MRN]" in redacted


def test_zip4_redacted():
    text = "Address is 90210-1234, near the clinic."
    redacted, _manifest = apply_rule_based_redaction(text)
    assert "90210-1234" not in redacted
    assert "[ZIP4]" in redacted


def test_dob_redacted():
    text = "DOB 05/17/1985 was on the intake form."
    redacted, manifest = apply_rule_based_redaction(text)
    assert "05/17/1985" not in redacted
    assert "[DOB]" in redacted
    assert any(m["type"] == "DOB" for m in manifest)


def test_insurance_id_redacted():
    text = "Insurance ID: M1234567890 on file."
    redacted, _manifest = apply_rule_based_redaction(text)
    assert "M1234567890" not in redacted
    assert "[INSURANCE_ID]" in redacted


def test_manifest_reports_original_offsets():
    text = "MRN: 123456 has SSN 999-88-7777."
    _redacted, manifest = apply_rule_based_redaction(text)
    types = {m["type"] for m in manifest}
    assert types == {"MRN", "SSN"}
    mrn = next(m for m in manifest if m["type"] == "MRN")
    ssn = next(m for m in manifest if m["type"] == "SSN")
    assert text[mrn["start"]:mrn["end"]] == "MRN: 123456"
    assert text[ssn["start"]:ssn["end"]] == "999-88-7777"


def test_clean_text_untouched():
    text = "The nurse was kind and the room stayed clean."
    redacted, manifest = apply_rule_based_redaction(text)
    assert redacted == text
    assert manifest == []