import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms.functional as TF


def DoubleConv2D(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))

def copy_and_crop(input_tensors, target_tensors):
    target_size = target_tensors.size()[2]
    input_size = input_tensors.size()[2]

    required = input_size-target_size 
    delta = required//2 
    return input_tensors[:, :, delta:input_size-delta, delta:input_size-delta]

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Contracting Path or Down Parts
        for feature in features:
            self.downs.append(DoubleConv2D(in_channels, feature))
            in_channels = feature 
            
         #for feature in reversed(features):
        for feature in features[::-1]:
            self.ups.append(
                nn.ConvTranspose2d(
                feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(
                DoubleConv2D(feature*2, feature)
            )

        self.bottleneck = DoubleConv2D(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.max_pool(x) 

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape!= skip_connection.shape:
                #x = TF.resize(x, size=skip_connection.shape[2:])
                x = copy_and_crop(x, skip_connection)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    

x = torch.randn((3, 1, 160, 160)) 
model = Unet(in_channels=1, out_channels=1)
y = model(x) 
print(x.size())
print(y.size())
