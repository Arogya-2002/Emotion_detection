from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from src.pipeline.emotion_pipeline import EmotionDetectionPipeline
from src.exceptions import CustomException
from src.logger import logging

app = FastAPI(title="Emotion Detection API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify frontend URL like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = EmotionDetectionPipeline()


@app.post("/predict-emotion/")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        results = pipeline.run(image)
        results_dict = [r.__dict__ for r in results]

        return JSONResponse(content={"status": "success", "results": results_dict}, status_code=200)

    except CustomException as ce:
        logging.error(f"CustomException: {str(ce)}")
        return JSONResponse(content={"status": "error", "message": str(ce)}, status_code=500)

    except Exception as e:
        logging.error(f"Unhandled Exception: {str(e)}", exc_info=True)
        return JSONResponse(content={"status": "error", "message": "Internal server error"}, status_code=500)
