import os
import random

from typing import Text, List

import pydantic
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import (
    HumanMessage,
    BaseOutputParser
)

from exceptions import ActionFailed

OPENAI_API_KEY = random.choice(os.getenv('OPENAI_API_KEY', '').split(','))


class CustomStripOutputParser(BaseOutputParser):
    @property
    def _type(self) -> str:
        return 'string'

    def get_format_instructions(self) -> str:
        return 'delete enter'

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip()


class OpenAIProxy:
    def __init__(self, openai_api_key: str = None, **kwargs):
        try:
            self.llm = OpenAI(openai_api_key=openai_api_key or OPENAI_API_KEY,
                              **kwargs)
        except pydantic.error_wrappers.ValidationError as e:
            raise ActionFailed(retcode=-1, message=f'null api key, {e}')

    def text_cut(func):
        def wrapper(self, *args, **kwargs):
            try:
                result = func(self, *args, **kwargs)
            except Exception as e:
                raise ActionFailed(retcode=-1, message=f'openai.error, {e}')
            if isinstance(result, str):
                return result.strip()
            return result
        return wrapper

    @text_cut
    def predict(self, text: Text):
        return self.llm.predict(text)

    def predict_messages(self, texts: List[Text]):
        messages = [HumanMessage(content=text) for text in texts]
        return self.llm.predict_messages(messages)


class TranslateAIChain:
    SYSTEM_TEMPLATE = """
    You are a helpful assistant who translate korea to simplified chinese.
A user will pass in a korea text, and you should translate it to simplified chinese, and nothing more.
    """
    HUMAN_TEMPLATE = prompt = """
    {text}
    translate to chinese simplified
    """

    def __init__(self, **kwargs):
        self.chain = LLMChain(
            llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, **kwargs),
            prompt=self.construct_prompt(),
            output_parser=CustomStripOutputParser()
        )

    def construct_prompt(self) -> ChatPromptTemplate:
        system_message_prompt = SystemMessagePromptTemplate.from_template(self.SYSTEM_TEMPLATE)
        human_message_prompt = HumanMessagePromptTemplate.from_template(self.HUMAN_TEMPLATE)
        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    def run(self, text: Text):
        return self.chain.run(text)
