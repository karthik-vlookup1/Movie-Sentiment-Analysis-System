from pydantic import BaseModel, Field, field_validator


class ReviewRequest(BaseModel):
    """Request body accepted by the prediction endpoint."""

    review: str = Field(..., description="Movie review to classify")

    @field_validator("review")
    @classmethod
    def review_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Review must not be empty.")
        return value.strip()


class PredictionResponse(BaseModel):
    review: str
    sentiment: str
    confidence: float
