from pydantic import BaseModel, Field
from typing import Optional, List


class FilteredSound(BaseModel):
    """Individual filtered sound entry from LLM response."""

    speech_index: int = Field(description="Index of the sentence")
    speech_text: str = Field(description="Original text of the sentence")
    should_add_sound: bool = Field(description="Whether a sound should be added")
    target_word: Optional[str] = Field(
        default=None,
        description="Specific word where to place the sound (null if should_add_sound=false)",
    )
    selected_sound_index: int = Field(
        description="Index (0, 1, or 2) of the selected sound"
    )
    reasoning: str = Field(description="Explanation of the decision")
    relevance_rank: int = Field(description="Unique integer rank (1 = most relevant)")


class FilterResponse(BaseModel):
    """Complete LLM filter response structure."""

    filtered_sounds: List[FilteredSound] = Field(
        description="List of all filtered sounds"
    )
