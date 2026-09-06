import uuid
from typing import List, Optional
from pydantic import BaseModel, Field
class Fact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    source_episode_id: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    also_seen_in: List[str] = Field(default_factory=list)