import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os
from datetime import datetime
from huggingface_hub import snapshot_download

# Create directories
CACHE_DIR = "model_cache"
OUTPUT_DIR = "generated_images"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def check_gpu():
    """Check GPU availability and return status"""
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            return True
        return False
    except Exception as e:
        st.error(f"Error checking GPU: {str(e)}")
        return False

def save_image(image, prompt):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Added milliseconds
    safe_prompt = "".join(x for x in prompt[:30] if x.isalnum() or x in (' ','-','_')).strip()
    filename = f"{timestamp}_{safe_prompt}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    image.save(filepath)
    return filepath

def download_model(model_id):
    """Download model once and cache it"""
    local_path = os.path.join(CACHE_DIR, model_id.split('/')[-1])
    
    if not os.path.exists(local_path):
        st.info(f"Downloading model {model_id} (this will happen only once)...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_path,
            local_dir_use_symlinks=False
        )
    return local_path

# Add these models to the available options
MODELS = {
    "Stable Diffusion 2.1": "stabilityai/stable-diffusion-2-1",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Realistic Vision": "SG161222/Realistic_Vision_V4.0",
    "OpenJourney": "prompthero/openjourney",
    "Dreamlike Diffusion": "dreamlike-art/dreamlike-diffusion-1.0"
}

# Add more image size presets with higher resolutions
IMAGE_SIZES = {
    "Square (512x512)": {
        "width": 512,
        "height": 512,
        "description": "Standard square format"
    },
    "Square HD (1024x1024)": {
        "width": 1024,
        "height": 1024,
        "description": "High resolution square format"
    },
    "Rectangular 16:9 (912x512)": {
        "width": 912,
        "height": 512,
        "description": "Widescreen format"
    },
    "Portrait HD (1080x1920)": {
        "width": 1080,
        "height": 1920,
        "description": "Mobile/Portrait HD format"
    },
    "Landscape HD (1920x1080)": {
        "width": 1920,
        "height": 1080,
        "description": "Landscape HD format"
    }
}

# Modify load_model function to handle different models
@st.cache_resource
def load_model(model_id):
    try:
        local_path = download_model(model_id)
        
        # Model-specific parameters
        model_params = {
            "stabilityai/stable-diffusion-xl-base-1.0": {
                "torch_dtype": torch.float16,
                "use_safetensors": True,
                "variant": "fp16"
            },
            "default": {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "safety_checker": None,
                "requires_safety_checker": False
            }
        }
        
        # Get model specific params or default
        params = model_params.get(model_id, model_params["default"])
        
        pipe = StableDiffusionPipeline.from_pretrained(
            local_path,
            local_files_only=True,
            **params
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
            pipe.enable_xformers_memory_efficient_attention()
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Update generation parameters for high-res images
def generate_image(pipe, prompt, model_id, size_preset):
    try:
        # Model-specific parameters with high-res adjustments
        generation_params = {
            "stabilityai/stable-diffusion-xl-base-1.0": {
                "num_inference_steps": 50,  # Increased for better quality
                "guidance_scale": 9.0,
                "width": IMAGE_SIZES[size_preset]["width"],
                "height": IMAGE_SIZES[size_preset]["height"]
            },
            "default": {
                "num_inference_steps": 50,
                "guidance_scale": 8.5,
                "width": IMAGE_SIZES[size_preset]["width"],
                "height": IMAGE_SIZES[size_preset]["height"]
            }
        }
        
        # Add warning for high-res images
        if IMAGE_SIZES[size_preset]["width"] * IMAGE_SIZES[size_preset]["height"] > 512 * 512:
            st.warning("Generating high-resolution image. This may take longer and require more GPU memory.")
        
        enhanced_prompt = f"high quality, detailed, 4k, {prompt}"
        negative_prompt = "ugly, blurry, bad quality, dark, low resolution, deformed, distorted"
        
        # Get model specific params or default
        params = generation_params.get(model_id, generation_params["default"])
        
        with torch.inference_mode():
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                image = pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    **params  # Now params is defined
                ).images[0]
                
                if image is None or image.getextrema() == (0, 0) or image.getextrema() == (255, 255):
                    raise Exception("Generated image is invalid")
                
                return image
    except Exception as e:
        st.error(f"Error during image generation: {str(e)}")
        return None

# Update main function's model selection
def main():
    st.title("AI Image Generator")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if check_gpu():
        st.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
        st.info(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        st.warning("No GPU detected. Using CPU (this will be slower)")
    
    # Create columns for input options
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.selectbox(
            "Select Model",
            list(MODELS.keys()),
            help="Different models specialize in different types of images"
        )
    
    with col2:
        size_preset = st.selectbox(
            "Image Size",
            list(IMAGE_SIZES.keys()),
            help="Choose the aspect ratio and size of the generated image"
        )
    
    prompt = st.text_area(
        "Enter your prompt (be descriptive):", 
        help="Example: A beautiful mountain landscape at sunset, with realistic details, 4k quality"
    )
    
    model_id = MODELS[model_name]
    
    # Display selected size info
    st.info(f"Selected size: {IMAGE_SIZES[size_preset]['width']}x{IMAGE_SIZES[size_preset]['height']} - {IMAGE_SIZES[size_preset]['description']}")
    
    if st.button("Generate Image"):
        if not prompt or len(prompt.strip()) < 3:
            st.warning("Please enter a detailed prompt (at least 3 characters)!")
            return
            
        with st.spinner("Generating image... This may take a minute..."):
            try:
                # Clear GPU memory before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                pipe = load_model(model_id)
                if pipe is None:
                    return
                
                image = generate_image(pipe, prompt, model_id, size_preset)
                if image is None:
                    return
                
                filepath = save_image(image, prompt)
                st.image(image, caption=f"Generated Image for: {prompt}", use_container_width=True)
                
                with open(filepath, "rb") as file:
                    btn = st.download_button(
                        label="Download Image",
                        data=file,
                        file_name=os.path.basename(filepath),
                        mime="image/png"
                    )
                
                st.success(f"Image saved to: {filepath}")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                # Clean up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

def download_all_models():
    models = [
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "SG161222/Realistic_Vision_V4.0",
        "prompthero/openjourney",
        "dreamlike-art/dreamlike-diffusion-1.0"
    ]
    
    for model_id in models:
        try:
            download_model(model_id)
            st.success(f"Downloaded: {model_id}")
        except Exception as e:
            st.error(f"Failed to download {model_id}: {str(e)}")

if __name__ == "__main__":
    if st.button("Download All Models"):
        download_all_models()
        st.success("All models downloaded successfully!")
    main()