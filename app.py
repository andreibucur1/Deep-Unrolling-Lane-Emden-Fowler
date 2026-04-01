import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io
import cv2
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from model import LEFNet, get_potentials

st.set_page_config(page_title="LEFNet Denoising AI", layout="wide")
st.title("Image Restoration: LEFNet vs Classical Methods")
st.markdown("Hybrid framework based on PDE equations (LEFNet + HJB) compared with baseline filters (NLM and Median).")


@st.cache_resource
def load_ai_model():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LEFNet(No_iterations=15).to(dev)
    try:
        net.load_state_dict(torch.load("lefnet_bsds300_global.pth", map_location=dev))
        net.eval()
        return net, dev, True
    except FileNotFoundError:
        return None, dev, False


model_lefnet, device, model_loaded = load_ai_model()

if not model_loaded:
    st.error("Could not find the file 'lefnet_bsds300_global.pth'. Make sure it is in the same folder as app.py!")
    st.stop()


@st.cache_data
def apply_nlm_uint8(img_uint8):
    bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    denoised_bgr = cv2.fastNlMeansDenoisingColored(bgr, None, h=15, hColor=15, templateWindowSize=7,
                                                   searchWindowSize=21)
    return cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)


@st.cache_data
def apply_median_uint8(img_uint8, ksize=5):
    bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    median_bgr = cv2.medianBlur(bgr, ksize=ksize)
    return cv2.cvtColor(median_bgr, cv2.COLOR_BGR2RGB)


@st.cache_data
def run_hjb_filter(image_np, iterations, dt=0.05):
    img_f = image_np.astype(np.float32) / 255.0
    for _ in range(iterations):
        for c in range(img_f.shape[2]):
            u = img_f[:, :, c]
            dy, dx = np.gradient(u)
            dyy, dyx = np.gradient(dy)
            dxy, dxx = np.gradient(dx)

            grad_mag = np.sqrt(dx ** 2 + dy ** 2 + 1e-8)

            u_nn = (dx ** 2 * dxx + 2 * dx * dy * dxy + dy ** 2 * dyy) / (grad_mag ** 2)

            u -= dt * np.sign(u_nn) * grad_mag
            img_f[:, :, c] = u

    return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)


st.sidebar.header("Control Parameters")

uploaded_file = st.sidebar.file_uploader("1. Upload an image", type=['png', 'jpg', 'jpeg'])
sigma_val = st.sidebar.slider("2. Gaussian Noise Level (Sigma)", 0.0, 0.20, 0.09, 0.01)
hjb_iters = st.sidebar.slider("3. HJB Iterations (Sharpness)", 0, 50, 10, 1)

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert('RGB')
    transform = transforms.ToTensor()
    img_clean_tensor = transform(img_pil).unsqueeze(0).to(device)

    torch.manual_seed(42)
    noise = sigma_val * torch.randn_like(img_clean_tensor)
    img_noisy_tensor = torch.clamp(img_clean_tensor + noise, 0.0, 1.0)
    noisy_np = (img_noisy_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    with st.spinner("Processing classical methods (NLM & Median)..."):
        nlm_np = apply_nlm_uint8(noisy_np)
        median_np = apply_median_uint8(noisy_np, ksize=5)

    with st.spinner("Processing PDE equation (LEFNet)..."):
        _, _, H, W = img_noisy_tensor.shape
        p_pot, q_pot = get_potentials(H, W, device)

        with torch.no_grad():
            img_lefnet_tensor = model_lefnet(img_noisy_tensor, p_pot, q_pot)
            img_lefnet_tensor = torch.clamp(img_lefnet_tensor, 0.0, 1.0)

        lefnet_np = (img_lefnet_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    with st.spinner("Applying HJB shock filter..."):
        if hjb_iters > 0:
            final_np = run_hjb_filter(lefnet_np, hjb_iters)
        else:
            final_np = lefnet_np


    st.markdown("### Visual Comparison")

    # Row 1
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Noisy (Original)")
        st.image(noisy_np, use_container_width=True)
    with col2:
        st.subheader("2. Median Filter")
        st.image(median_np, use_container_width=True)
    with col3:
        st.subheader("3. NLM (Non-Local Means)")
        st.image(nlm_np, use_container_width=True)

    st.markdown("---")

    # Row 2
    col4, col5, col6 = st.columns(3)
    with col4:
        st.subheader("4. LEFNet (Denoising)")
        st.image(lefnet_np, use_container_width=True)
    with col5:
        st.subheader("5. LEFNet + HJB (Final)")
        st.image(final_np, use_container_width=True)
    with col6:
        st.subheader(" Save Result")
        st.markdown("Download the enhanced version obtained with LEFNet + HJB.")

        result_pil = Image.fromarray(final_np)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="⬇️ Download Final Image",
            data=byte_im,
            file_name="result_LEFNet_HJB.png",
            mime="image/png",
            use_container_width=True
        )
else:
    st.info("Please upload an image from the sidebar to start processing.")