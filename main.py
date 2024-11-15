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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(x for x in prompt[:30] if x.isalnum() or x in (' ','-','_')).strip()
    filename = f"{timestamp}_{safe_prompt}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    image.save(filepath)
    return filepath

# Update this list with the RobMix Zenith identifier
model_option = st.selectbox(
    "Select Model",
    ["stabilityai/stable-diffusion-2-1", "civitai/robmix-zenith"]
)

def download_model(model_id):
    """Download model once and cache it"""
    local_path = os.path.join(CACHE_DIR, model_id.split('/')[-1])
    
    # Ensure compatibility with both Hugging Face and CivitAI identifiers
    if model_id == "civitai/robmix-zenith":
        # Assuming you've downloaded RobMix Zenith manually or via a CivitAI downloader
        st.info(f"Using locally available RobMix Zenith model: {local_path}")
    elif not os.path.exists(local_path):
        st.info(f"Downloading model {model_id} (this will happen only once)...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_path,
            local_dir_use_symlinks=False
        )
    return local_path


@st.cache_resource
def load_model(model_id):
    try:
        local_path = download_model(model_id)
        
        pipe = StableDiffusionPipeline.from_pretrained(
            local_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True,
            safety_checker=None,  # Disable safety checker
            requires_safety_checker=False
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
            pipe.enable_xformers_memory_efficient_attention()  # Enable memory efficient attention
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def generate_image(pipe, prompt):
    try:
        # Enhance the prompt
        enhanced_prompt = f"high quality, detailed, 4k, {prompt}"
        negative_prompt = "ugly, blurry, bad quality, dark, low resolution, deformed, distorted"
        
        with torch.inference_mode():
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                image = pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=50,  # Increased steps
                    guidance_scale=8.5,      # Increased guidance
                    width=512,
                    height=512,
                ).images[0]
                
                # Verify image generation
                if image is None or image.getextrema() == (0, 0) or image.getextrema() == (255, 255):
                    raise Exception("Generated image is invalid")
                
                return image
    except Exception as e:
        st.error(f"Error during image generation: {str(e)}")
        return None

def main():
    st.title("AI Image Generator")
    
    # Add memory management at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if check_gpu():
        st.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
        st.info(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        st.warning("No GPU detected. Using CPU (this will be slower)")
    
    prompt = st.text_area("Enter your prompt (be descriptive):", 
                         help="Example: A beautiful mountain landscape at sunset, with realistic details, 4k quality")
        
    if st.button("Generate Image"):
        if not prompt or len(prompt.strip()) < 3:
            st.warning("Please enter a detailed prompt (at least 3 characters)!")
            return
            
        with st.spinner("Generating image... This may take a minute..."):
            try:
                # Clear GPU memory before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                pipe = load_model(model_option)
                if pipe is None:
                    return
                
                image = generate_image(pipe, prompt)
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
        # "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1"
    ]
    
    for model_id in models:
        download_model(model_id)

if __name__ == "__main__":
    if st.button("Download All Models"):
        download_all_models()
        st.success("All models downloaded successfully!")
    main()