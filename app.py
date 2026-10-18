import os
import io
import gc
import math
import zipfile
import requests
import streamlit as st
from PIL import Image
import numpy as np
import torch
from spandrel import ModelLoader

st.set_page_config(page_title="Universal AI Upscaler", layout="wide")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

AUTO_MODELS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "4x-UltraSharp.pth": "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth",
    "4x_NMKD-Superscale-SP_178000_G.pth": "https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth",
}


@st.cache_resource(max_entries=1)
def load_ai_model(model_name):
    filepath = os.path.join(MODELS_DIR, model_name)
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)

    if os.path.exists(root_path):
        filepath = root_path
    elif not os.path.exists(filepath) and model_name in AUTO_MODELS:
        with st.spinner(f"Downloading {model_name} to cloud server..."):
            r = requests.get(AUTO_MODELS[model_name], stream=True)
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    loader = ModelLoader(torch.device("cpu"))
    descriptor = loader.load_from_file(filepath)
    descriptor.model.eval()
    return descriptor


def get_available_models():
    models = list(AUTO_MODELS.keys())
    for folder in [MODELS_DIR, os.path.dirname(os.path.abspath(__file__))]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith((".pth", ".safetensors")) and f not in models:
                    models.append(f)
    return models


def tile_inference(img_tensor, descriptor, tile_size=256, tile_pad=16):
    b, c, h, w = img_tensor.shape
    scale = descriptor.scale
    raw_model = descriptor.model
    output = torch.zeros((b, c, h * scale, w * scale), dtype=torch.float32)

    tiles_x = math.ceil(w / tile_size)
    tiles_y = math.ceil(h / tile_size)
    total_tiles = tiles_x * tiles_y
    current = 0

    bar = st.progress(0, text="Processing tiles...")
    for y in range(tiles_y):
        for x in range(tiles_x):
            current += 1
            bar.progress(current / total_tiles, text=f"Processing tile {current} of {total_tiles}...")

            ofs_x, ofs_y = x * tile_size, y * tile_size
            in_x1, in_x2 = ofs_x, min(ofs_x + tile_size, w)
            in_y1, in_y2 = ofs_y, min(ofs_y + tile_size, h)

            pad_x1, pad_x2 = max(in_x1 - tile_pad, 0), min(in_x2 + tile_pad, w)
            pad_y1, pad_y2 = max(in_y1 - tile_pad, 0), min(in_y2 + tile_pad, h)

            tile = img_tensor[:, :, pad_y1:pad_y2, pad_x1:pad_x2]
            with torch.no_grad():
                out_tile = raw_model(tile)

            out_x1, out_x2 = in_x1 * scale, in_x2 * scale
            out_y1, out_y2 = in_y1 * scale, in_y2 * scale

            tx1 = (in_x1 - pad_x1) * scale
            tx2 = tx1 + (in_x2 - in_x1) * scale
            ty1 = (in_y1 - pad_y1) * scale
            ty2 = ty1 + (in_y2 - in_y1) * scale

            output[:, :, out_y1:out_y2, out_x1:out_x2] = out_tile[:, :, ty1:ty2, tx1:tx2]

    bar.empty()
    return output


def upscale_pil(orig_pil, descriptor, scale_factor):
    has_alpha = orig_pil.mode == "RGBA"
    rgb_img = orig_pil.convert("RGB")
    alpha = orig_pil.split()[3] if has_alpha else None

    img_np = np.array(rgb_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    out_tensor = tile_inference(img_tensor, descriptor)
    out_np = (out_tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    upscaled_pil = Image.fromarray(out_np)

    target_w = int(round(orig_pil.width * scale_factor))
    target_h = int(round(orig_pil.height * scale_factor))
    if upscaled_pil.size != (target_w, target_h):
        upscaled_pil = upscaled_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)

    if has_alpha:
        upscaled_pil.putalpha(alpha.resize((target_w, target_h), Image.Resampling.LANCZOS))

    del img_tensor, out_tensor, out_np
    gc.collect()
    return upscaled_pil, target_w, target_h


# ================= SIDEBAR GLOBAL CONTROLS =================
with st.sidebar:
    st.header("⚙️ Engine Settings")
    model_choice = st.selectbox("AI Model", get_available_models())
    scale_factor = st.slider("Upscale Factor (1.0x - 4.0x)", 1.0, 4.0, 2.0, 0.05)
    out_format = st.selectbox("Output Format", ["PNG", "WEBP", "JPG"])

    st.divider()
    custom_pth = st.file_uploader("Upload Custom Model (.pth / .safetensors)", type=["pth", "safetensors"])
    if custom_pth:
        save_path = os.path.join(MODELS_DIR, custom_pth.name)
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(custom_pth.getbuffer())
            st.success(f"Added {custom_pth.name}! Select it in the dropdown above.")
            st.rerun()

st.title("🚀 Universal AI Upscaler (Forge GAN Engine)")
tab_single, tab_batch = st.tabs(["🖼️ Single Image", "📁 Batch Images"])

# ================= TAB 1: SINGLE IMAGE =================
with tab_single:
    col1, col2 = st.columns(2)
    with col1:
        single_file = st.file_uploader("Select Input Image", type=["png", "jpg", "jpeg", "webp"], key="single")
        if single_file:
            orig_pil = Image.open(single_file)
            st.image(orig_pil, caption=f"Original ({orig_pil.width}x{orig_pil.height} px)", use_container_width=True)
            run_single_btn = st.button("✨ Upscale Image", type="primary", use_container_width=True)

    with col2:
        if single_file and run_single_btn:
            descriptor = load_ai_model(model_choice)
            upscaled_pil, tw, th = upscale_pil(orig_pil, descriptor, scale_factor)
            st.image(upscaled_pil, caption=f"Upscaled ({tw}x{th} px)", use_container_width=True)

            buf = io.BytesIO()
            fmt_save = "JPEG" if out_format == "JPG" else out_format
            if fmt_save == "JPEG":
                upscaled_pil.convert("RGB").save(buf, format=fmt_save, quality=95)
            else:
                upscaled_pil.save(buf, format=fmt_save)

            ext = out_format.lower()
            st.download_button(
                f"💾 Download Upscaled .{ext.upper()}",
                data=buf.getvalue(),
                file_name=f"upscaled_{scale_factor}x.{ext}",
                mime=f"image/{ext}",
                use_container_width=True
            )

# ================= TAB 2: BATCH IMAGES =================
with tab_batch:
    batch_files = st.file_uploader(
        "Select Multiple Images to Batch Upscale",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="batch"
    )
    if batch_files:
        st.info(f"Loaded {len(batch_files)} images ready for batch processing.")
        if st.button("🚀 Start Batch Upscale", type="primary"):
            descriptor = load_ai_model(model_choice)
            zip_buffer = io.BytesIO()
            overall_bar = st.progress(0, text="Starting batch...")
            ext = out_format.lower()
            fmt_save = "JPEG" if out_format == "JPG" else out_format

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, b_file in enumerate(batch_files, start=1):
                    overall_bar.progress(idx / len(batch_files), text=f"Batch Progress: Image {idx} of {len(batch_files)} ({b_file.name})")
                    img = Image.open(b_file)
                    up_pil, _, _ = upscale_pil(img, descriptor, scale_factor)

                    img_buf = io.BytesIO()
                    if fmt_save == "JPEG":
                        up_pil.convert("RGB").save(img_buf, format=fmt_save, quality=95)
                    else:
                        up_pil.save(img_buf, format=fmt_save)

                    base_name = os.path.splitext(b_file.name)[0]
                    zf.writestr(f"{base_name}_upscaled_{scale_factor}x.{ext}", img_buf.getvalue())

            overall_bar.empty()
            st.success(f"Successfully upscaled {len(batch_files)} images!")
            st.download_button(
                "📦 Download All Upscaled Images (.ZIP)",
                data=zip_buffer.getvalue(),
                file_name="upscaled_batch.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )
