"""fact tuple schema: what the extractor produces from one episode"""
from pydantic import BaseModel,Field
from typing import Optional

class Fact(BaseModel):
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    source_episode_id: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)



    