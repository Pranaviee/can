import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from models_mae import mae_vit_base_patch16
import numpy as np
from torchvision.utils import save_image

import os

# ---- Config ----
img_path = 'sample3.JPEG'  # Update with your image path
checkpoint_path = './can_out/checkpoint-99.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Load Model ----
model = mae_vit_base_patch16(norm_pix_loss=True, noise_loss=True, std=0.1)
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# ---- Transform ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- Load Image and Create Two Views ----
img = Image.open(img_path).convert('RGB')
img_tensor_1 = transform(img).to(device)
img_tensor_2 = transform(img).to(device)
img_tensor = torch.stack([img_tensor_1, img_tensor_2], dim=0).to(device)  # Shape: [2, 3, 224, 224]
img_tensor_1=img_tensor_1.unsqueeze(0)
patches=model.patchify(img_tensor_1)
print("Patch shape:",patches.shape)
patchify_reccon=model.unpatchify(patches).clamp(0,1)
save_image(patchify_reccon,"results/step1_patchified.jpg")
# ---- Forward Pass ----
with torch.no_grad():
    loss, loss_contrastive, loss_noise, pred, mask = model(img_tensor, mask_ratio=0.25)
    mask_img1=mask[0]
    patches_img1=model.patchify(img_tensor_1)
    visible=patches_img1.clone()
    visible[0][mask_img1.bool()]=0
    visible_img=model.unpatchify(visible).clamp(0,1)
    save_image(visible_img,"results/step2_visible_masked_input.jpg")
    reconstructed = model.unpatchify(pred)  # shape: [2, 3, 224, 224]
    reconstructed = torch.clamp(reconstructed, 0, 1)

# ---- Unnormalize for Display ----
def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
    return img_tensor * std + mean

img_orig = torch.clamp(unnormalize(img_tensor), 0, 1)

# ---- Plot First View and Its Reconstruction ----
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_orig[0].permute(1, 2, 0).cpu().numpy())
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(reconstructed[0].permute(1, 2, 0).cpu().numpy())
plt.axis('off')

plt.tight_layout()
plt.show()

# ---- Save Latent Features (Optional) ----
with torch.no_grad():
    latent, _, _ = model.forward_encoder(img_tensor, mask_ratio=0.75)
latent_np = latent[0].detach().cpu().numpy()
np.save("latent_view1.npy", latent_np)

print("✅ Reconstruction and latent features complete. Saved latent_view1.npy")
