from unet import Unet 
import torch 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
from tqdm import tqdm 
import torch.nn as nn 
import torch.optim as optim
from utils import load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predicstions_as_imgs

learning_rate = 1e-4 
device = "cuda" if torch.cuda.is_available() else "cpu" 
batch_size = 32 
num_epochs = 10
num_workers = 2 
image_height = 160 
image_width = 240 
PIN_MEMORY = True 
LOAD_MODEL = False 

train_img_dir = "data/train"
train_mask_dir = "data/train_masks"
val_img_dir = "data/val" 
val_mask_dir = "data/val_masks" 

def train(loader, model, optimizer, loss, scaler):
    loop = tqdm(loader)  

    for batch_idx, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        #training 
        with torch.cuda.amp.autocast():
            predictions = model(images) 
            losses = loss(predictions, labels)

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer) 
        scaler.update()

        #update tqdm loop 
        loop.set_postfix(loss=losses.item())

        
        

def main():
    train_transforms = A.Compose(
        [
        A.Resize(height=image_height, width=image_width),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0], 
            std = [1.0, 1.0, 1.0], 
            max_pixel_value=255.0
        ),
        ToTensorV2(),
          
        ]
    ) 
    val_transforms = A.Compose(
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

    model = Unet(in_channels=3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss() #cross entropy for multiclass
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = get_loaders(
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        batch_size,
        train_transforms,
        val_transforms
    )

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        train(train_loader, model, optimizer, loss_fn, scaler) 
        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint) 

        check_accuracy(val_loader, model, device=device)

        save_predicstions_as_imgs(val_loader, model, folder="saved_images/", device=device)
        

if __name__ == "__main__":
    main()

