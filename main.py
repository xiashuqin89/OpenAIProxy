import os

import uvicorn
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import models
import auth
import api
import multi
from exceptions import ActionFailed


oauth2_scheme = auth.SelfOAuth2PasswordBearer(tokenUrl="token")
app = FastAPI(dependencies=[Depends(oauth2_scheme)])
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(os.getenv('ALLOW_ORIGINS', '')),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ActionFailed)
async def unicorn_exception_handler(request: Request, exc: ActionFailed):
    return JSONResponse(
        status_code=418,
        content={
            'code': exc.retcode,
            'result': False,
            'data': None,
            'message': exc.message
        }
    )


@app.get("/")
async def root():
    return "Hello world!"


@app.post("/openai/predict/")
async def openai_predict(item: models.OpenAI):
    answer = api.OpenAIProxy(item.api_token, **(item.model_config or {})).predict(item.text)
    return {
        'code': 0,
        'result': True,
        'message': '',
        'data': {'answer': answer}
    }


@app.post("/openai/multi-predict/")
async def openai_predict(item: models.MultiAI):
    ai = multi.MultiAIChain(item.prompt,
                            item.chunk_size or 200,
                            item.chunk_overlap or 0,
                            item.delay or 0.5)
    answer = ai.run(item.text)
    return {
        'code': 0,
        'result': True,
        'message': '',
        'data': {'answer': answer}
    }



if __name__ == '__main__':
    uvicorn.run(app="main:app", host="0.0.0.0", port=8000, reload=True)

