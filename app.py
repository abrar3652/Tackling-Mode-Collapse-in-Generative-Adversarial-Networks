import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os

# ====================== Page Config ======================
st.set_page_config(
    page_title="Anime Face GAN Demo",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Anime Face Generator")
st.markdown("**DCGAN vs WGAN-GP** — Tackling Mode Collapse")

# ====================== Device ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.success(f"Device: **{device}**")

# ====================== Generator Architecture ======================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# ====================== Load Models from Checkpoint ======================
@st.cache_resource
def load_generator(checkpoint_path, model_name):
    if not os.path.exists(checkpoint_path):
        st.error(f"❌ Checkpoint not found: {checkpoint_path}")
        st.stop()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract netG from checkpoint
    if 'netG' in checkpoint:
        state_dict = checkpoint['netG']
    else:
        state_dict = checkpoint  # fallback if only state_dict was saved

    model = Generator().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    st.sidebar.success(f"✅ {model_name} loaded successfully!")
    return model

# Load both models
dcgan_model = load_generator("dcgan_checkpoint.pth", "DCGAN")
wgan_model  = load_generator("wgangp_checkpoint.pth", "WGAN-GP")

# ====================== Preprocessing & Generation ======================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def generate_images(generator, num_samples=4, latent_dim=100):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim, 1, 1, device=device)
        fake_imgs = generator(noise)
        fake_imgs = (fake_imgs * 0.5 + 0.5).clamp(0, 1)
        fake_imgs = fake_imgs.cpu().permute(0, 2, 3, 1).numpy()
    return (fake_imgs * 255).astype(np.uint8)

# ====================== Sidebar ======================
st.sidebar.header("Generation Settings")
model_choice = st.sidebar.radio("Select Model", ["DCGAN", "WGAN-GP"], index=1)
num_samples = st.sidebar.slider("Number of Images", min_value=1, max_value=8, value=4)

# ====================== Main App ======================
if st.button("🎲 Generate New Anime Faces", type="primary"):
    with st.spinner("Generating..."):
        selected_model = dcgan_model if model_choice == "DCGAN" else wgan_model
        images = generate_images(selected_model, num_samples=num_samples)

        cols = st.columns(num_samples)
        for i, img_array in enumerate(images):
            pil_img = Image.fromarray(img_array)
            with cols[i]:
                st.image(pil_img, caption=f"Sample {i+1}", use_column_width=True)

st.info("""
**Note:**  
- WGAN-GP usually produces better and more diverse results than DCGAN.  
- Images are generated at **64×64** resolution (as trained).  
- This demo uses the models trained in the "Tackling Mode Collapse" notebook.
""")

st.caption("Streamlit App for Tackling Mode Collapse in GANs")