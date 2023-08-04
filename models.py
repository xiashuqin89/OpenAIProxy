from typing import Union, Dict

from pydantic import BaseModel


class OpenAI(BaseModel):
    text: str
    api_token: Union[str, None] = None
    model_config: Union[Dict, None] = None


class MultiAI(BaseModel):
    text: str
    prompt: str
    chunk_size: Union[int, None] = None
    chunk_overlap: Union[int, None] = None
    delay: Union[float, None] = None
