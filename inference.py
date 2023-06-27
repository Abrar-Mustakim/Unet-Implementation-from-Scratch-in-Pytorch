import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms.functional as TF
from unet import Unet
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt

def infer_and_visualize(model, image_path, mask_path, transform, device="cuda", figname="inference_result.png"):
  image = np.array(Image.open(image_path).convert("RGB"))
  mask = np.array(Image.open(mask_path).convert("L"))
  img = transform(image=image)
  print(img)
  img = img['image']
  img = img.to(device)
  image1 = torch.unsqueeze(img, 0)
  image1 = image1.to(device)
  prediction = torch.sigmoid(model(image1)) 
  prediction = (prediction > 0.5).float() 
  prediction = prediction.squeeze(0).cpu().detach().numpy()
  output = np.transpose(prediction, (1, 2, 0)) 

  fig, axs = plt.subplots(1, 3, figsize=(15, 10))
  axs[0].imshow(image)
  axs[0].set_title('Actual Image')
  #axs[1].imshow(mask, cmap='gray')
  axs[1].imshow(mask)
  axs[1].set_title('Actual Mask')
  #axs[2].imshow(output, cmap='gray')
  axs[2].imshow(output)
  axs[2].set_title('Predicted Mask')
  plt.tight_layout()
  plt.savefig(figname)
  plt.show()
  



device = "cuda" if torch.cuda.is_available() else "cpu" 
image_height = 160 
image_width = 240 
model = Unet(in_channels=3, out_channels=1).to(device)

checkpoint = torch.load("my_checkpoint.pth.tar", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)

transforms = A.Compose(
        [
        A.Resize(height=image_height, width=image_width),
        A.Normalize(
            mean=[0.0, 0.0, 0.0], 
            std = [1.0, 1.0, 1.0], 
            max_pixel_value=255.0
        ),
        ToTensorV2(),
          
        ]
    )


img_path = "data/val/1b25ea8ba94d_13.jpg"
mask_path = "data/val_masks/1b25ea8ba94d_13_mask.gif"
infer_and_visualize(model, img_path, mask_path, transforms, device="cuda", figname="Inference Result.png")



