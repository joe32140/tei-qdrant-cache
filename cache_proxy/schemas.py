from pydantic import BaseModel, Field
from typing import List, Union, Optional

class EmbedRequest(BaseModel):
    inputs: Union[str, List[str]]
    # Add other potential TEI parameters if needed, e.g., truncate: bool = False
    normalize: Optional[bool] = None # Pass through common params
    prompt_name: Optional[str] = None
    prompt: Optional[str] = None

# We expect TEI to return a list of lists (embeddings)
class TEIEmbedResponse(BaseModel):
    embeddings: List[List[float]]
    # Include other fields if TEI adds them, like usage stats

# Using Any to simplify, TEI might return List[List[float]] or Dict with embeddings
class GenericTEIResponse(BaseModel):
     embeddings: List[List[float]] = Field(alias="result") # Adapt based on actual TEI response structure if needed