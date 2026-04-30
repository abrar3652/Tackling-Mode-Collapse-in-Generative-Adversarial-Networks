import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os

# ====================== Page Config ======================
st.set_page_config(
    page_title="Doodle to Real - Colorization",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Doodle to Real Image Translation & Colorization")
st.markdown("Upload a **sketch/doodle** (black lines on white or simple drawing) and get a realistic colored version.")

# ====================== Device ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.success(f"Running on: **{device}**")

# ====================== Generator Model ======================
class UNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(input_nc, ngf, normalize=False)      # 256->128
        self.enc2 = self._conv_block(ngf, ngf*2)                         # 128->64
        self.enc3 = self._conv_block(ngf*2, ngf*4)                       # 64->32
        self.enc4 = self._conv_block(ngf*4, ngf*8)                       # 32->16
        self.enc5 = self._conv_block(ngf*8, ngf*8)                       # 16->8
        self.enc6 = self._conv_block(ngf*8, ngf*8)                       # 8->4
        self.enc7 = self._conv_block(ngf*8, ngf*8)                       # 4->2
        self.enc8 = self._conv_block(ngf*8, ngf*8, normalize=False)      # 2->1

        # Decoder
        self.dec1 = self._deconv_block(ngf*8, ngf*8, dropout=True)
        self.dec2 = self._deconv_block(ngf*8*2, ngf*8, dropout=True)
        self.dec3 = self._deconv_block(ngf*8*2, ngf*8, dropout=True)
        self.dec4 = self._deconv_block(ngf*8*2, ngf*4)
        self.dec5 = self._deconv_block(ngf*4*2, ngf*2)
        self.dec6 = self._deconv_block(ngf*2*2, ngf)
        self.dec7 = self._deconv_block(ngf*2, ngf)
        self.dec8 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _conv_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _deconv_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # Decoder with skip connections
        d1 = self.dec1(e8)
        d2 = self.dec2(torch.cat([d1, e7], 1))
        d3 = self.dec3(torch.cat([d2, e6], 1))
        d4 = self.dec4(torch.cat([d3, e5], 1))
        d5 = self.dec5(torch.cat([d4, e4], 1))
        d6 = self.dec6(torch.cat([d5, e3], 1))
        d7 = self.dec7(torch.cat([d6, e2], 1))
        d8 = self.dec8(torch.cat([d7, e1], 1))

        return d8


# ====================== Load Model ======================
@st.cache_resource
def load_generator():
    model = UNetGenerator(input_nc=3, output_nc=3, ngf=64).to(device)
    
    checkpoint_path = "G_epoch45.pth"
    if not os.path.exists(checkpoint_path):
        st.error(f"❌ Model file `{checkpoint_path}` not found in the current directory.")
        st.stop()
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    st.sidebar.success("✅ Generator model loaded successfully!")
    return model

generator = load_generator()

# ====================== Image Preprocessing ======================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    np_img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_img)

# ====================== Sidebar ======================
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Generation Strength (optional)", 0.5, 2.0, 1.0, 0.1)

# ====================== Main App ======================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Doodle / Sketch")
    uploaded_file = st.file_uploader("Choose a sketch image...", type=["png", "jpg", "jpeg", "webp"])
    
    if uploaded_file:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Input Sketch", use_column_width=True)

with col2:
    st.subheader("Generated Real Image")
    if uploaded_file and st.button("🎨 Generate Realistic Image", type="primary"):
        with st.spinner("Generating... This may take a few seconds on CPU"):
            try:
                input_tensor = preprocess_image(input_image)
                
                with torch.no_grad():
                    output_tensor = generator(input_tensor)
                
                output_image = tensor_to_image(output_tensor)
                
                st.image(output_image, caption="Generated Real Image", use_column_width=True)
                
                # Download button
                buf = io.BytesIO()
                output_image.save(buf, format="PNG")
                buf.seek(0)
                
                st.download_button(
                    label="⬇️ Download Generated Image",
                    data=buf,
                    file_name="generated_real_image.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"Error during generation: {e}")

# ====================== Instructions ======================
st.info("""
**Tips for best results:**
- Use black lines on white/transparent background
- Simple, clean sketches work better (especially faces or anime-style)
- Image will be automatically resized to 256×256
- Model works best with sketches similar to CUHK or Anime sketch dataset
""")

st.caption("Built with Streamlit • Powered by Pix2Pix-style U-Net Generator (G_epoch45.pth)")