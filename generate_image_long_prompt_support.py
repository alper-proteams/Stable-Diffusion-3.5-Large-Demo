import torch
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig
from diffusers import SD3Transformer2DModel

# Model ID
model_id = "stabilityai/stable-diffusion-3.5-large"

# Configure 4-bit quantization for lower VRAM usage
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model with quantization
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

# Create the pipeline
pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=model_nf4,
    torch_dtype=torch.bfloat16
)

# Enable CPU offloading for better memory management
pipeline.enable_model_cpu_offload()

# Your prompt
prompt = "Mouthwatering triple-layer dark chocolate cake on a vintage mint green cake stand, rich chocolate ganache dripping down sides, fudgy chocolate buttercream swirls with visible texture, topped with fresh chocolate shavings and maraschino cherries, 1950s pastel blue and yellow kitchen background with checkerboard linoleum floor, classic Formica countertop, retro mint green refrigerator, chrome details, nostalgic afternoon sunlight streaming through ruffled gingham curtains, soft shadows, old-fashioned copper cookware visible in background, professional food photography, 85mm lens, f/2.8, extreme close-up details of moist cake crumb structure, condensation on cherries, photorealistic, cinematic color grading, slight film grain, Kodak Portra 400 look, 8k resolution"

# Generate image with recommended parameters
image = pipeline(
    prompt=prompt,
    num_inference_steps=28,  # Recommended value from model card
    guidance_scale=4.5,      # Recommended value
    max_sequence_length=512  # Allows for longer prompts
).images[0]

# Save the image
image.save("cake.png")