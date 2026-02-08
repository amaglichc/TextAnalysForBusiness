from pydantic import BaseModel, Field

class TextRequest(BaseModel):
    text: str = Field(..., max_length=1000)

class TopicResponse(BaseModel):
    topic_id: int
    topic_name: str
    confidence: float
    
    
class AnalysisResponse(BaseModel):
    sentiment: str
    topic: TopicResponse
    summary: str