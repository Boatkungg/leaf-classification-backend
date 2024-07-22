from fastapi import FastAPI, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from fastai.learner import load_learner
import io
from typing import Annotated
import platform

import pathlib

plt = platform.system()
if plt == "Linux":
    pathlib.WindowsPath = pathlib.PosixPath

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your pre-trained Fastai model (replace with your path)
learn_inf = load_learner("Main_model2.pkl")
check_leaf = load_learner("good_or_bad.pkl")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
async def predict(image: Annotated[bytes, File()]):
    predicted_class = None
    probability = None

    try:
        image = Image.open(io.BytesIO(image)).resize((224, 224))

        image = np.array(image.convert("RGB"))

        if check_leaf.predict(image)[0] == "good":

            # Make prediction using the loaded learner
            prediction = learn_inf.predict(image)

            # Extract class and probability from the prediction
            predicted_class = prediction[0]
            probability = prediction[1].item()

            return {
                "class": predicted_class,
                "probability": probability,
            }
        else:
            return {
                "class": "bad",
                "probability": 0.0,
            }

    except Exception as e:
        print("Error = ", e)
        raise HTTPException(
            status_code=500, detail="An error occurred during prediction"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)  # Change port if needed
