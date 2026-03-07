from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

import model as m
import agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    m.load_model()
    yield


app = FastAPI(title="roomaudit", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post("/inspect")
async def inspect(file: UploadFile):
    if m.model is None:
        raise HTTPException(503, "Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    result = agent.inspect_with_agent(image)
    return result
