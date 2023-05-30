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


app = FastAPI()
templates = Jinja2Templates(directory="html-templates")
progress_event = asyncio.Event()


class CsvInfo(BaseModel):
    row_count: int
    column_count: int
    column_names: List[str]
    histogram_image: str


def process_csv(csv_file, strict):
    # Send progress update (e.g., 30%)
    app.progress = 30
    progress_event.set()

    # Read .csv file using pandas
    df = pd.read_csv(io.StringIO(csv_file.decode("utf-8")))

    # Process the .csv file (here, we're just getting basic info as an example)
    row_count = len(df)
    column_count = len(df.columns)
    column_names = df.columns.tolist()

    # Create a histogram using seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df)
    plt.title("Histogram of CSV Values")

    # Save the histogram to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Encode the image as base64
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Send progress update (e.g., 60%)
    app.progress = 60
    progress_event.set()

    if strict:
        # Perform additional processing or validation if required
        pass

    # Send progress update (e.g., 100%)
    app.progress = 100
    progress_event.set()

    return {
        "row_count": row_count,
        "column_count": column_count,
        "column_names": column_names,
        "histogram_image": image_base64,
    }


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
