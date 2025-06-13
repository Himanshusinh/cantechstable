from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
import torch
from diffusers import StableDiffusion3Pipeline
import os

# Initialize FastAPI app
app = FastAPI()

# ✅ Replace with your actual frontend domain
origins = [
    "https://wildmindai.com",  # Replace with your production frontend domain
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the model ID
model_id = "stabilityai/stable-diffusion-3.5-medium"

# Create output directory if it doesn't exist
os.makedirs("generated", exist_ok=True)

# Load Stable Diffusion 3.5 pipeline without BitsAndBytesConfig
pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()

# Define request schema
class PromptRequest(BaseModel):
    prompt: str

# POST endpoint to generate image
@app.post("/generate")
async def generate_image(req: PromptRequest):
    filename = f"{uuid4().hex}.png"
    image = pipeline(
        prompt=req.prompt,
        num_inference_steps=60,
        guidance_scale=5.5,
    ).images[0]

    filepath = f"generated/{filename}"
    image.save(filepath)

    return {"url": f"/images/{filename}"}

# GET endpoint to retrieve image
@app.get("/images/{filename}")
async def get_image(filename: str):
    filepath = f"generated/{filename}"
    if os.path.exists(filepath):
        return FileResponse(path=filepath, media_type="image/png")
    return {"error": "Image not found"}
