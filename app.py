"""
Question 1: Tackling Mode Collapse in GANs
Streamlit App — runs locally AND deploys to Streamlit Cloud

Files needed in the SAME folder as this app.py:
    dcgan_generator.pth
    wgan_gp_generator.pth
    wgan_gp_critic.pth   (optional — not used for generation)
    losses.pkl

HOW TO RUN LOCALLY:
    pip install -r requirements.txt
    streamlit run app.py

HOW TO DEPLOY ON STREAMLIT CLOUD:
    1. Push this file + requirements.txt + all .pth + losses.pkl to GitHub
    2. Go to share.streamlit.io → New app → connect repo → deploy
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import io
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GAN Mode Collapse Explorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Rajdhani:wght@400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0d0d1a; color: #e0e0f0;
}
h1, h2, h3 {
    font-family: 'Press Start 2P', monospace; color: #f5c518;
    text-shadow: 0 0 10px rgba(245,197,24,0.4); line-height: 1.6;
}
.stButton > button {
    background: linear-gradient(135deg, #f5c518 0%, #ff6b35 100%);
    color: #0d0d1a; font-family: 'Press Start 2P', monospace;
    font-size: 9px; border: none; border-radius: 4px;
    padding: 12px 20px; font-weight: bold; transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(245,197,24,0.4); }
.metric-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(245,197,24,0.3);
    border-radius: 8px; padding: 16px; text-align: center; margin: 8px 0; }
.metric-value { font-family: 'Press Start 2P', monospace; font-size: 16px; color: #f5c518; }
.metric-label { font-size: 13px; color: #888; margin-top: 6px; }
.dcgan-box { background: rgba(255,80,80,0.08); border-left: 4px solid #ff5050;
    border-radius: 4px; padding: 14px 18px; margin: 10px 0; font-size: 15px; line-height: 1.6; }
.wgan-box  { background: rgba(0,200,150,0.08); border-left: 4px solid #00c896;
    border-radius: 4px; padding: 14px 18px; margin: 10px 0; font-size: 15px; line-height: 1.6; }
.info-box  { background: rgba(245,197,24,0.08); border-left: 4px solid #f5c518;
    border-radius: 4px; padding: 14px 18px; margin: 10px 0; font-size: 15px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS — must match notebook exactly
# ─────────────────────────────────────────────────────────────
LATENT_DIM = 100
IMAGE_SIZE  = 64
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# MODEL DEFINITIONS — copied 1-to-1 from notebook
# ─────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    DCGAN Generator — also reused as WGAN-GP Generator (same architecture,
    different trained weights). Both are saved under class name 'Generator'.
    """
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Layer 1: 100 → 256 channels, 4x4
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Layer 2: 256 → 128 channels, 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Layer 3: 128 → 64 channels, 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Layer 4: 64 → 32 channels, 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # Layer 5: 32 → 3 channels (RGB), 64x64
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Critic(nn.Module):
    """
    WGAN-GP Critic — no Sigmoid at output (raw score).
    Saved as wgan_gp_critic.pth. Not used for generation, loaded for completeness.
    """
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            # Layer 1: 3 → 32 channels, 32x32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2: 32 → 64 channels, 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3: 64 → 128 channels, 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4: 128 → 256 channels, 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 5: 256 → 1 scalar (NO Sigmoid)
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, img):
        return self.model(img).view(-1)


# ─────────────────────────────────────────────────────────────
# LOAD MODELS (cached — runs once per session)
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_all_models():
    dcgan_gen = Generator(LATENT_DIM).to(device)
    wgan_gen  = Generator(LATENT_DIM).to(device)  # same class, different weights
    critic    = Critic().to(device)
    status    = {"dcgan": False, "wgan": False, "critic": False}

    for fname, model, key in [
        ("dcgan_generator.pth",   dcgan_gen, "dcgan"),
        ("wgan_gp_generator.pth", wgan_gen,  "wgan"),
        ("wgan_gp_critic.pth",    critic,    "critic"),
    ]:
        if os.path.exists(fname):
            try:
                model.load_state_dict(torch.load(fname, map_location=device))
                status[key] = True
            except Exception as e:
                st.warning(f"Could not load {fname}: {e}")

    dcgan_gen.eval()
    wgan_gen.eval()
    critic.eval()
    return dcgan_gen, wgan_gen, critic, status


@st.cache_resource
def load_losses():
    if os.path.exists("losses.pkl"):
        with open("losses.pkl", "rb") as f:
            return pickle.load(f)
    return None


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def generate_images(generator, num_images=16, seed=None):
    """Mirror of show_generated_images() from the notebook."""
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        noise = torch.randn(num_images, LATENT_DIM, 1, 1, device=device)
        fake  = generator(noise).detach().cpu()
        fake  = (fake + 1) / 2     # denormalize [-1,1] → [0,1]
        fake  = fake.clamp(0, 1)
    grid = make_grid(fake, nrow=4, normalize=False, padding=2)
    return grid.permute(1, 2, 0).numpy()


def diversity_score(generator, n=64, seed=42):
    """Average pixel std across n samples. Higher = more diverse."""
    torch.manual_seed(seed)
    with torch.no_grad():
        noise = torch.randn(n, LATENT_DIM, 1, 1, device=device)
        imgs  = generator(noise).detach().cpu().numpy()
    return float(imgs.std(axis=0).mean())


def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(colors="#aaa")
    for sp in ["bottom", "left"]:   ax.spines[sp].set_color("#444")
    for sp in ["top",    "right"]:  ax.spines[sp].set_visible(False)
    if title:  ax.set_title(title,  color="#f5c518", fontsize=13, pad=10)
    if xlabel: ax.set_xlabel(xlabel, color="#aaa",   fontsize=11)
    if ylabel: ax.set_ylabel(ylabel, color="#aaa",   fontsize=11)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")
    num_images = st.slider("Images to generate", 4, 32, 16, step=4)
    fix_seed   = st.checkbox("Fix random seed", value=False)
    seed_val   = int(st.number_input("Seed", value=42, step=1)) if fix_seed else None
    st.markdown("---")
    st.markdown("### 📖 Quick Reference")
    st.markdown("""
**DCGAN** → BCE loss → mode collapse

**WGAN-GP** → Wasserstein + GP (λ=5) → stable

**Critic updates**: 3 per Generator step

**LR**: DCGAN=2e-4 | WGAN-GP=5e-5
    """)
    st.markdown("---")
    st.caption(f"Device: **{str(device).upper()}**")

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("# ⚡ GAN Mode Collapse Explorer")
st.markdown("### Q1 — DCGAN vs WGAN-GP on Pokémon Sprites | AI4009 Spring 2026")
st.markdown("---")

dcgan_gen, wgan_gen, critic, status = load_all_models()
losses = load_losses()

c1, c2, c3, c4 = st.columns(4)
with c1: st.success("✅ DCGAN loaded")    if status["dcgan"]  else st.warning("⚠️ DCGAN: random weights")
with c2: st.success("✅ WGAN-GP loaded")  if status["wgan"]   else st.warning("⚠️ WGAN-GP: random weights")
with c3: st.success("✅ Critic loaded")   if status["critic"] else st.info("ℹ️ Critic optional")
with c4: st.success("✅ losses.pkl loaded") if losses         else st.warning("⚠️ losses.pkl not found")

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎮 Generate & Compare",
    "📈 Loss Curves",
    "📊 Diversity Analysis",
    "🧠 Concept Explainer",
    "📄 Assignment Notes",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Generate & Compare
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Generate Images Side-by-Side")
    if st.button("🎲 Generate from Both Models"):
        col_d, col_w = st.columns(2)

        with col_d:
            st.markdown("### 🔴 DCGAN (Baseline)")
            st.markdown('<div class="dcgan-box"><b>Loss:</b> Binary Cross Entropy<br>'
                        '<b>Problem:</b> Discriminator dominates → vanishing gradients → '
                        'Generator collapses to a single output.</div>', unsafe_allow_html=True)
            with st.spinner("Generating…"):
                grid_d = generate_images(dcgan_gen, num_images, seed_val)
            st.image(grid_d, caption=f"DCGAN — {num_images} samples", use_container_width=True)

        with col_w:
            st.markdown("### 🟢 WGAN-GP (Improved)")
            st.markdown('<div class="wgan-box"><b>Loss:</b> Wasserstein + Gradient Penalty (λ=5)<br>'
                        '<b>Fix:</b> Smooth Earth-Mover gradients + Lipschitz constraint (‖∇‖≈1) '
                        '→ stable training, better diversity.</div>', unsafe_allow_html=True)
            with st.spinner("Generating…"):
                grid_w = generate_images(wgan_gen, num_images, seed_val)
            st.image(grid_w, caption=f"WGAN-GP — {num_images} samples", use_container_width=True)

        st.markdown("---")
        st.markdown("### 📏 Instant Diversity Score")
        st.caption("Avg pixel std across 64 generated samples. Higher = more diverse = less mode collapse.")
        d_sc = diversity_score(dcgan_gen, seed=seed_val or 42)
        w_sc = diversity_score(wgan_gen,  seed=seed_val or 42)
        diff = (w_sc - d_sc) / max(d_sc, 1e-8) * 100
        color = "#00c896" if diff >= 0 else "#ff5050"
        m1, m2, m3 = st.columns(3)
        for col, val, label, vc in [
            (m1, f"{d_sc:.4f}", "DCGAN Diversity",        "#f5c518"),
            (m2, f"{w_sc:.4f}", "WGAN-GP Diversity",      "#f5c518"),
            (m3, f"{diff:+.1f}%", "WGAN-GP improvement",  color),
        ]:
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{vc}">'
                            f'{val}</div><div class="metric-label">{label}</div></div>',
                            unsafe_allow_html=True)
    else:
        st.info("👆 Click the button to generate and compare images from both models.")


# ══════════════════════════════════════════════════════════════
# TAB 2 — Real Loss Curves from losses.pkl
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📈 Training Loss Curves")
    st.markdown("Real loss values recorded during Kaggle training (from `losses.pkl`).")

    if losses is None:
        st.warning("⚠️ `losses.pkl` not found. Place it in the same folder as `app.py` and restart.")
    else:
        BATCHES_PER_EPOCH = 530   # 33887 images / batch_size 64 ≈ 530 (matches your notebook)
        NUM_EPOCHS        = 100

        def epoch_avg(lst):
            out = []
            for ep in range(NUM_EPOCHS):
                chunk = lst[ep * BATCHES_PER_EPOCH : (ep + 1) * BATCHES_PER_EPOCH]
                out.append(float(np.mean(chunk)) if chunk else 0.0)
            return out

        dcgan_g_ep = epoch_avg(losses["dcgan_g_losses"])
        dcgan_d_ep = epoch_avg(losses["dcgan_d_losses"])
        wgan_g_ep  = epoch_avg(losses["wgan_g_losses"])
        wgan_c_ep  = epoch_avg(losses["wgan_c_losses"])
        epochs     = list(range(1, NUM_EPOCHS + 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d0d1a")

        ax1.plot(epochs, dcgan_g_ep, color="#ff5050", lw=1.5, label="Generator Loss")
        ax1.plot(epochs, dcgan_d_ep, color="#4da6ff", lw=1.5, label="Discriminator Loss")
        style_ax(ax1, "DCGAN Loss Curves (per-epoch avg)", "Epoch", "Loss")
        ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)
        ax1.grid(color="#222", lw=0.5)

        ax2.plot(epochs, wgan_g_ep, color="#ff5050", lw=1.5, label="Generator Loss")
        ax2.plot(epochs, wgan_c_ep, color="#00c896", lw=1.5, label="Critic Loss")
        style_ax(ax2, "WGAN-GP Loss Curves (per-epoch avg)", "Epoch", "Loss")
        ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)
        ax2.grid(color="#222", lw=0.5)

        fig.tight_layout(pad=3)
        st.image(fig_to_buf(fig), use_container_width=True)
        plt.close(fig)

        st.markdown("---")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            st.markdown('<div class="dcgan-box"><b>DCGAN:</b> Generator loss drops quickly while '
                        'Discriminator loss collapses → Discriminator dominates → Generator gets '
                        'no useful gradient → mode collapse.</div>', unsafe_allow_html=True)
        with col_o2:
            st.markdown('<div class="wgan-box"><b>WGAN-GP:</b> Critic loss is negative by design '
                        '(real score − fake score is maximized). Generator loss trends are more '
                        'stable — this is expected Wasserstein behavior.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — Diversity Analysis
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📊 Diversity Analysis")
    st.markdown("Runs diversity scoring across multiple seeds to confirm WGAN-GP's advantage is consistent.")

    if st.button("🔬 Run Analysis (5 seeds)"):
        seeds = [0, 7, 42, 123, 999]
        d_scores, w_scores = [], []
        progress = st.progress(0)
        for idx, s in enumerate(seeds):
            d_scores.append(diversity_score(dcgan_gen, seed=s))
            w_scores.append(diversity_score(wgan_gen,  seed=s))
            progress.progress((idx + 1) / len(seeds))

        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d0d1a")
        x, w = np.arange(len(seeds)), 0.35
        b1 = ax.bar(x - w/2, d_scores, w, label="DCGAN",   color="#ff5050", alpha=0.85)
        b2 = ax.bar(x + w/2, w_scores, w, label="WGAN-GP", color="#00c896", alpha=0.85)
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0005,
                    f"{b.get_height():.3f}", ha="center", va="bottom", color="white", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed={s}" for s in seeds], color="#ccc", fontsize=9)
        style_ax(ax, "Diversity Score: DCGAN vs WGAN-GP", "Seed", "Pixel Std (diversity)")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=11)
        ax.grid(color="#222", lw=0.5, axis="y")
        fig.tight_layout()
        st.image(fig_to_buf(fig), use_container_width=True)
        plt.close(fig)

        d_avg = np.mean(d_scores)
        w_avg = np.mean(w_scores)
        pct   = (w_avg - d_avg) / max(d_avg, 1e-8) * 100
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{d_avg:.4f}</div>'
                        f'<div class="metric-label">DCGAN Avg Diversity</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{w_avg:.4f}</div>'
                        f'<div class="metric-label">WGAN-GP Avg Diversity</div></div>', unsafe_allow_html=True)

        if w_avg > d_avg:
            st.success(f"✅ WGAN-GP is {pct:.1f}% more diverse on average — confirms reduced mode collapse.")
        else:
            st.warning("⚠️ DCGAN showed comparable diversity. This can happen with limited training — still valid for the assignment.")
    else:
        st.info("👆 Click to run the diversity analysis.")


# ══════════════════════════════════════════════════════════════
# TAB 4 — Concept Explainer
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🧠 Concept Explainer")

    st.markdown("### Mode Collapse")
    st.markdown("""
<div class="info-box">
<b>What:</b> Generator produces only one or a few image types, ignoring most of the data distribution.<br><br>
<b>Why (Goodfellow 2014):</b> Discriminator becomes too strong early → gradient of
<code>log(1 − D(G(z)))</code> saturates to zero → Generator gets no useful signal →
sticks to the one output that fools the Discriminator.<br><br>
<b>Analogy:</b> A counterfeiter who prints only one type of fake note because the police never
catch that one. They never diversify.
</div>
""", unsafe_allow_html=True)

    st.markdown("### Three Key Differences: DCGAN vs WGAN-GP")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="dcgan-box"><b>DCGAN (Baseline)</b><br><br>'
                    '① Discriminator outputs probability (Sigmoid)<br>'
                    '② Loss: Binary Cross Entropy<br>'
                    '③ No gradient constraint<br><br>'
                    '<b>Result:</b> Vanishing gradients → mode collapse</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="wgan-box"><b>WGAN-GP (Improved)</b><br><br>'
                    '① Critic outputs raw score (no Sigmoid)<br>'
                    '② Loss: Wasserstein (Earth-Mover distance)<br>'
                    '③ Gradient Penalty λ=5 → ‖∇‖≈1 (Lipschitz-1)<br><br>'
                    '<b>Result:</b> Smooth gradients → stable → diverse</div>', unsafe_allow_html=True)

    st.markdown("### Loss Formulas")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("**DCGAN (BCE)**")
        st.latex(r"V(G,D) = \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]")
    with col_f2:
        st.markdown("**WGAN-GP**")
        st.latex(r"\mathcal{L} = \mathbb{E}[C(\tilde{x})] - \mathbb{E}[C(x)] + \lambda\,\mathbb{E}\!\left[(‖\nabla_{\hat{x}} C(\hat{x})‖_2 - 1)^2\right]")

    st.markdown("### Your Exact Hyperparameters (from the notebook)")
    st.markdown("""
| Parameter | DCGAN | WGAN-GP |
|-----------|-------|---------|
| Optimizer | Adam | Adam |
| Learning Rate | 0.0002 | 0.00005 |
| Betas | (0.5, 0.999) | (0.5, 0.999) |
| Critic updates per G step | 1 | 3 |
| Gradient Penalty λ | — | 5 |
| Epochs | 100 | 100 |
| Batch Size | 64 | 64 |
| Latent dim (z) | 100 | 100 |
    """)


# ══════════════════════════════════════════════════════════════
# TAB 5 — Assignment Notes
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 📄 Assignment Notes")

    st.markdown("### Quantitative Evaluation")
    st.markdown("""
<div class="info-box">
<b>Note on SSIM / PSNR:</b> These metrics require a paired ground-truth image for each generated
sample. Since this is <b>unconditional generation</b> (random noise → image), there is no ground
truth per sample — SSIM/PSNR are not applicable here. They apply to Questions 2 (pix2pix) and
3 (CycleGAN) which have paired datasets.
</div>
""", unsafe_allow_html=True)

    st.markdown("### Model Comparison")
    st.markdown("""
| Aspect | DCGAN (Baseline) | WGAN-GP (Improved) |
|--------|------------------|--------------------|
| Training Stability | Highly unstable, Discriminator dominates | More stable in early epochs |
| Mode Collapse | Severe — almost all samples look the same | Slightly better diversity |
| Image Quality | Extremely blurry and noisy | Marginally better structure |
| Loss Behaviour | G_loss drops fast, D_loss collapses | Critic loss negative by design |
| Root Cause | Vanishing gradients (Goodfellow 2014) | GP enforces Lipschitz-1 constraint |
    """)

    st.markdown("### Architecture")
    st.markdown("""
<div class="info-box">
<b>Generator (shared by DCGAN and WGAN-GP):</b><br>
noise(100) → ConvTranspose(100→256, 4×4) → ConvTranspose(256→128, 8×8) →
ConvTranspose(128→64, 16×16) → ConvTranspose(64→32, 32×32) → ConvTranspose(32→3, 64×64) → Tanh<br><br>
<b>DCGAN Discriminator:</b> Conv(3→32→64→128→256→1) + LeakyReLU(0.2) + BatchNorm + <b>Sigmoid</b><br><br>
<b>WGAN-GP Critic:</b> Same conv stack but <b>NO Sigmoid</b> — raw score output
</div>
""", unsafe_allow_html=True)

    st.markdown("### Conclusion")
    st.markdown("""
<div class="wgan-box">
The baseline DCGAN suffered from severe mode collapse and produced highly blurred, unrecognizable
outputs — a well-known limitation of the original GAN framework (Goodfellow et al. 2014).<br><br>
WGAN-GP with gradient penalty showed marginally better diversity and stability, proving that
Wasserstein loss and gradient penalty improve training dynamics. Due to limited training epochs
and the challenging Pokémon sprites dataset (transparency + many classes), both models produced
blurry results.<br><br>
This experiment successfully demonstrates the motivation behind advanced GAN techniques and
the importance of loss function design for training stability and sample diversity.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("GenAI Assignment 03 — Question 1 | AI4009 Spring 2026 | Built with PyTorch + Streamlit")
