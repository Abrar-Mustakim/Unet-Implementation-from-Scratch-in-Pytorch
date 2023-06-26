import random 
import os 
import shutil
#a = [3, 4, 5, 6, 7, 8, 9, 10]
#random.shuffle(a) 
#print(a)


train_dir = os.path.join("train") 

train_lists = os.listdir("train") 
mask_lists = os.listdir("train_masks")
print(train_lists[:5])
#print(mask_lists[:5])
#random.shuffle(train_lists)
#val_images = random.sample(train_lists, 88)


combined = list(zip(train_lists, mask_lists))
random.shuffle(combined)

val_image_1, val_masks_1 = zip(*combined)

random_indices = random.sample(range(len(val_image_1)), 88)

val_images = [val_image_1[i] for i in random_indices]
val_masks =  [val_masks_1[i] for i in random_indices]

print(val_images[:5]) 
print(val_masks[:5])

for i in val_images:
    shutil.move("train/"+i, "val/"+i)

for i in val_masks:
    shutil.move("train_masks/"+i, "val_masks/"+i)

