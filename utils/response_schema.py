from utils.custom_imports import *

class ResponseSchema(BaseModel):
    """
    Structured schema for the agent's response
    """
    answer: str = Field(description="Comprehensive answer to the query")
    sources: List[Dict[str, str]] = Field(
        description="List of sources used to generate the answer",
        default_factory=list
    )
    confidence_score: float = Field(
        description="Confidence of the answer (0-1 scale)",
        ge=0, 
        le=1, 
        default=0.7
    )
    key_points: List[str] = Field(
        description="Key points extracted from the research",
        default_factory=list
    )

    @validator('sources', pre=True, always=True)
    def validate_sources(cls, v):
        # Ensure sources is a list of dictionaries
        if not v:
            return []
        return [{'source': src} if isinstance(src, str) else src for src in v]

    @validator('confidence_score', pre=True, always=True)
    def validate_confidence(cls, v):
        # Ensure confidence is between 0 and 1
        try:
            return max(0, min(float(v), 1))
        except (TypeError, ValueError):
            return 0.5