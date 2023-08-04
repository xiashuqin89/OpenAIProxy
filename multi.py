import os
import time
import itertools
from typing import Dict
from multiprocessing.dummy import Pool as ThreadPool

from langchain.text_splitter import RecursiveCharacterTextSplitter

import api
import exceptions


OPENAI_API_KEYS = os.getenv('OPENAI_API_KEY', '').split(',')

_openai_api_key_cache = {}


class MultiAIChain:
    def __init__(self,
                 prompt: str,
                 chunk_size=200,
                 chunk_overlap=0,
                 delay=0.5):
        self.prompt = prompt
        self.delay = delay
        self.openai_api_key_len = len(OPENAI_API_KEYS)
        self.pool = ThreadPool(self.openai_api_key_len - 1)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        self.cache = None

    def handler(self, element: Dict):
        ai = api.OpenAIProxy(openai_api_key=element['openai_api_key'])
        try:
            answer = ai.predict(self.prompt.format(element['text']))
        except exceptions.ActionFailed as e:
            return {
                'index': element['index'],
                'answer': f"{element['text']}[undo]"
            }
        time.sleep(self.delay)
        return {
            'index': element['index'],
            'answer': answer
        }

    def _extend_openai_api_key(self, segment_len: int):
        multiple = round(segment_len / self.openai_api_key_len) + 1
        for i in range(multiple):
            yield OPENAI_API_KEYS

    def _mapper(self, document: str):
        segments = self.splitter.split_text(document)
        openai_api_keys = list(itertools.chain(*self._extend_openai_api_key(len(segments))))
        segments = [
            {
                'index': seg[0],
                'text': seg[1],
                'openai_api_key':openai_api_keys.pop()
            } for seg in enumerate(segments)
        ]
        self.cache = self.pool.map(self.handler, segments, self.openai_api_key_len - 1)

    def _combiner(self):
        if self.cache:
            self.cache = sorted(self.cache, key=lambda x: x['index'])
            self.cache = [item['answer'] for item in self.cache]

    def _reducer(self):
        self.pool.close()
        self.pool.join()
        return ' '.join(self.cache)

    def run(self, document: str):
        self._mapper(document)
        self._combiner()
        return self._reducer()
