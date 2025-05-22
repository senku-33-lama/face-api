from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from deepface import DeepFace
import shutil
import os

app = FastAPI()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    temp_file = "temp.jpg"

    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = DeepFace.find(img_path=temp_file, db_path="known_faces", enforce_detection=False)
        if len(result) > 0 and not result[0].empty:
            name = result[0].iloc[0]['identity'].split("/")[-1].split(".")[0]
            return JSONResponse({"match": name})
        else:
            return JSONResponse({"match": "Unknown"})

    except Exception as e:
        return JSONResponse({"error": str(e)})

    finally:
        os.remove(temp_file)
