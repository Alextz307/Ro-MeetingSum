from pydantic import BaseModel, Field


class DialogueTurn(BaseModel):
    """Represents a single speaker's turn in a debate."""

    speaker: str
    text: str


class MeetingSession(BaseModel):
    """Schema for a scraped parliamentary session."""

    session_id: int
    date: str

    # The 'Reference' summary (from the agenda/sumar)
    summary_points: list[str] = Field(default_factory=list)

    # The 'Input' text (the full transcript)
    full_transcript: str

    # Structured input for advanced models (PEGASUS)
    dialogue: list[DialogueTurn] = Field(default_factory=list)
