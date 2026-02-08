from pydantic import BaseModel, Field

class TextRequest(BaseModel):
    text: str = Field(..., max_length=1000)