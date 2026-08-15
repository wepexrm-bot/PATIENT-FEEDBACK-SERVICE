from pydantic import BaseModel, Field, field_validator

VALID_SOURCES = {"portal", "sms", "kiosk"}


class FeedbackIn(BaseModel):
    patient_ref: str = Field(..., min_length=1, description="Opaque token reference for the patient")
    text: str = Field(..., min_length=1, description="Raw free-text feedback")
    source: str = Field(..., pattern="^(portal|sms|kiosk)$")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in VALID_SOURCES:
            raise ValueError(f"source must be one of {sorted(VALID_SOURCES)}")
        return v


class FeedbackOut(BaseModel):
    model_config = {"protected_namespaces": ()}
    label: str
    confidence: float
    model_version: str
    redacted_text: str