import logging
import time
from typing import Callable

import uvicorn
from fastapi import FastAPI, Request

from routers.embeddings import router

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = FastAPI()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Callable):
    """All responses come with process time information"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8088, reload=True)
