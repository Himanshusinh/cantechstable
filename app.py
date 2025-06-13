from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
import torch
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
import os

# Initialize FastAPI app
app = FastAPI()

# ✅ Replace with your actual frontend domain from IONOS
origins = [
    "https://wildmindai.com",  # e.g., "https://wildmindai.com"
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

# Create a directory for generated images if not exists
os.makedirs("generated", exist_ok=True)

# Load quantized model configuration
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load the transformer model
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.float16
)

# Load the full SD3.5 pipeline
pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=model_nf4,
    torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()

# Request body structure
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

# GET endpoint to retrieve the generated image
@app.get("/images/{filename}")
async def get_image(filename: str):
    filepath = f"generated/{filename}"
    if os.path.exists(filepath):
        return FileResponse(path=filepath, media_type="image/png")
    return {"error": "Image not found"}
