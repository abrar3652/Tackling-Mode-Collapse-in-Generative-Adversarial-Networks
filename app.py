import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import gradio as gr
from PIL import Image

# ─── Model Architecture ───────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    def forward(self, x): return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, num_residual_blocks=6):
        super(Generator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x): return self.model(x)


# ─── Load Models from Combined File ──────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt = torch.load('cyclegan_final_generators.pth', map_location=DEVICE)

def load_state(model, state):
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model

G_AB = Generator().to(DEVICE)   # sketch -> photo
G_BA = Generator().to(DEVICE)   # photo  -> sketch

# try common key names used when saving combined checkpoints
if 'G_AB' in ckpt:
    load_state(G_AB, ckpt['G_AB'])
    load_state(G_BA, ckpt['G_BA'])
elif 'G_AB_state_dict' in ckpt:
    load_state(G_AB, ckpt['G_AB_state_dict'])
    load_state(G_BA, ckpt['G_BA_state_dict'])
elif 'netG_AB' in ckpt:
    load_state(G_AB, ckpt['netG_AB'])
    load_state(G_BA, ckpt['netG_BA'])
else:
    # print available keys to help debug
    print('Available keys in checkpoint:', list(ckpt.keys()))
    raise KeyError('Could not find G_AB/G_BA keys. Check printed keys above.')

print('CycleGAN models loaded successfully.')


# ─── Transforms ───────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def tensor_to_pil(tensor):
    img = tensor.squeeze(0).cpu()
    img = img * 0.5 + 0.5
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


# ─── Inference ────────────────────────────────────────────────────────────────

def translate(input_img, direction):
    img = input_img.convert('RGB')
    inp = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if direction == 'Sketch -> Photo':
            out = G_AB(inp)
        else:
            out = G_BA(inp)

    return tensor_to_pil(out)


# ─── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title='CycleGAN - Sketch to Photo Translation') as demo:
    gr.Markdown('# CycleGAN — Sketch ↔ Photo Translation')
    gr.Markdown(
        'Translate between sketch and photo domains using CycleGAN. '
        'Upload a sketch to generate a photo, or upload a photo to generate a sketch.'
    )

    direction = gr.Radio(
        ['Sketch -> Photo', 'Photo -> Sketch'],
        value='Sketch -> Photo',
        label='Translation Direction'
    )

    with gr.Row():
        inp_img = gr.Image(type='pil', label='Input Image')
        out_img = gr.Image(type='pil', label='Output Image')

    btn = gr.Button('Translate', variant='primary')
    btn.click(fn=translate, inputs=[inp_img, direction], outputs=out_img)

if __name__ == '__main__':
    demo.launch()