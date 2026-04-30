import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import gradio as gr
from PIL import Image

# ─── Model Architecture ───────────────────────────────────────────────────────

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


# ─── Load Models ──────────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

netG_DC = Generator().to(DEVICE)
ckpt_dc = torch.load('dcgan_checkpoint.pth', map_location=DEVICE)
netG_DC.load_state_dict(ckpt_dc['netG'])
netG_DC.eval()
print('DCGAN Generator loaded.')

netG_WGAN = Generator().to(DEVICE)
ckpt_wgan = torch.load('wgangp_checkpoint.pth', map_location=DEVICE)
netG_WGAN.load_state_dict(ckpt_wgan['netG'])
netG_WGAN.eval()
print('WGAN-GP Generator loaded.')


# ─── Inference ────────────────────────────────────────────────────────────────

def tensor_to_pil_grid(fake_imgs, nrow=5):
    grid    = vutils.make_grid(fake_imgs, nrow=nrow, padding=2, normalize=True)
    grid_np = np.transpose(grid.cpu().numpy(), (1, 2, 0))
    grid_np = (grid_np * 255).astype(np.uint8)
    return Image.fromarray(grid_np)


def generate(num_images, model_choice, seed):
    torch.manual_seed(int(seed))
    num_images = int(num_images)
    noise = torch.randn(num_images, 100, 1, 1, device=DEVICE)

    with torch.no_grad():
        fake_dc   = netG_DC(noise).cpu()
        fake_wgan = netG_WGAN(noise).cpu()

    dc_img   = tensor_to_pil_grid(fake_dc,   nrow=num_images)
    wgan_img = tensor_to_pil_grid(fake_wgan, nrow=num_images)

    if model_choice == 'DCGAN':
        return dc_img, None
    elif model_choice == 'WGAN-GP':
        return None, wgan_img
    else:
        return dc_img, wgan_img


# ─── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title='Anime Face Generation — DCGAN vs WGAN-GP') as demo:
    gr.Markdown('# Anime Face Generation — DCGAN vs WGAN-GP')
    gr.Markdown(
        'Generate anime faces using two GAN variants trained on the Anime Faces dataset. '
        'Compare DCGAN and WGAN-GP outputs side by side.'
    )

    with gr.Row():
        num_images   = gr.Slider(1, 10, value=5, step=1, label='Number of Images')
        model_choice = gr.Radio(['DCGAN', 'WGAN-GP', 'Both'], value='Both', label='Model')
        seed         = gr.Number(value=42, label='Seed')

    btn = gr.Button('Generate', variant='primary')

    with gr.Row():
        out_dc   = gr.Image(label='DCGAN Output')
        out_wgan = gr.Image(label='WGAN-GP Output')

    btn.click(
        fn=generate,
        inputs=[num_images, model_choice, seed],
        outputs=[out_dc, out_wgan]
    )

if __name__ == '__main__':
    demo.launch()