import logging
import time
from typing import Callable
import uvicorn
from fastapi import FastAPI, Request
from routers.embeddings import router
import argparse

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
    # create the command line argument parser
    parser = argparse.ArgumentParser(description="Run the embedding microservice")
    parser.add_argument("--host", default="0.0.0.0", type=str, help="host address to run on")
    parser.add_argument("--port", default=8089, type=int, help="port number to run on")
    
    # parse argument
    args = parser.parse_args()
    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)
