from typing import Union, Dict

from pydantic import BaseModel


class OpenAI(BaseModel):
    text: str
    api_token: Union[str, None] = None
    model_config: Union[Dict, None] = None


class Chain(BaseModel):
    text: str
