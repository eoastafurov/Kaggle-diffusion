from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import io
from fastapi.responses import FileResponse
import os
from fastapi.templating import Jinja2Templates
import io
import time
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import Request
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi import WebSocket
import asyncio
from fastapi import FastAPI, UploadFile, File, WebSocket
from processor import Processor

app = FastAPI()
templates = Jinja2Templates(directory="html-templates")
progress_event = asyncio.Event()


class CsvInfo(BaseModel):
    row_count: int
    mean_similarity: float
    std_similarity: float
    median_similarity: float
    intersection_percent: float
    sims_histplot: str


processor = Processor(
    path_to_gt="/home/toomuch/kaggle-diffusion/private-cv/gt_embbedings.parquet"
)


def process_csv(csv_file, strict):
    df = pd.read_csv(io.StringIO(csv_file.decode("utf-8")))

    # Send progress update (e.g., 100%)
    app.progress = 100
    progress_event.set()

    return processor(df)


@app.get("/download_test_indices")
async def download_test_indices():
    file_path = "test_indices.csv"
    if os.path.exists(file_path):
        return FileResponse(
            file_path, media_type="text/csv", filename="test_indices.csv"
        )
    else:
        return {"error": "File not found."}


@app.post("/process_csv", response_model=CsvInfo)
async def process_csv_endpoint(
    request: Request, file: UploadFile = File(...), strict: bool = Form(False)
):
    content = await file.read()
    csv_info = process_csv(content, strict)
    return templates.TemplateResponse("results.html", {"request": request, **csv_info})


@app.websocket("/progress")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        await progress_event.wait()
        progress = app.progress
        await websocket.send_text(str(progress))
        if progress >= 100:
            break
        progress_event.clear()


@app.get("/")
async def read_index():
    return FileResponse("index.html")
