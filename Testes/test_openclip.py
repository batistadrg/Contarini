import torch
import open_clip
from PIL import Image
import numpy as np
import pyautogui

# ============================
# CONFIGURAÇÕES
# ============================
modelo = "ViT-B-32"  # modelo leve
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega modelo e tokenizer
model, _, preprocess = open_clip.create_model_and_transforms(modelo, pretrained="laion2b_s34b_b79k")
model = model.to(device)
tokenizer = open_clip.get_tokenizer(modelo)

# ============================
# INPUT DO USUÁRIO
# ============================
texto_alvo = "copiar código"     # botão procurado

# Screenshot da tela inteira
img = pyautogui.screenshot()
img = img.convert("RGB")

# ============================
# DIVIDIR A IMAGEM EM PATCHES
# ============================
PATCH = 64  # tamanho do “bloco” escaneado

w, h = img.size

coords = []
patches = []

for y in range(0, h, PATCH):
    for x in range(0, w, PATCH):
        crop = img.crop((x, y, x + PATCH, y + PATCH))

        coords.append((x, y))
        patches.append(preprocess(crop))

# Empacota para tensor
image_input = torch.stack(patches).to(device)

# Texto → embedding
text_input = tokenizer([texto_alvo]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_input)

# Normaliza
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Similaridades (texto vs cada patch)
similaridades = (image_features @ text_features.T).squeeze()

# Pega o melhor patch
indice_melhor = torch.argmax(similaridades).item()
melhor_x, melhor_y = coords[indice_melhor]

# ============================
# RESULTADO
# ============================
print(f"\n=== RESULTADO ===")
print(f"Botão '{texto_alvo}' encontrado próximo de:")
print(f"X = {melhor_x}px,  Y = {melhor_y}px")
pyautogui.moveTo(melhor_x, melhor_y)
