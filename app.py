from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import FileResponse
import torch
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from uuid import uuid4

app = FastAPI()

model_id = "stabilityai/stable-diffusion-3.5-medium"

# Load model once at startup
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.float16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=model_nf4,
    torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_image(req: PromptRequest):
    filename = f"{uuid4().hex}.png"
    image = pipeline(
        prompt=req.prompt,
        num_inference_steps=60,
        guidance_scale=5.5,
    ).images[0]

    image.save(f"generated/{filename}")
    return {"url": f"/images/{filename}"}

@app.get("/images/{filename}")
async def get_image(filename: str):
    return FileResponse(path=f"generated/{filename}", media_type="image/png")
