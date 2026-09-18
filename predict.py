import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image

#image_paths = ['./img/guapo.jpeg', './img/burro.jpeg', './img/chocado.jpeg', './img/dormindo.jpeg', './img/olhando.jpeg']
image_paths = ['./img/1.jpeg', './img/2.jpeg', './img/3.jpeg', './img/4.jpeg', './img/5.jpeg', 
               './img/6.jpeg', './img/7.jpeg', './img/8.jpeg', './img/9.jpeg', './img/10.jpeg',
                 './img/11.jpeg', './img/12.jpeg', './img/messi.jpeg']

images = [load_image(img) for img in image_paths]

torch.Size([3,32,32])
class_names = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

class NeuralNet(nn.Module):

    def __init__ (self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,12,5) # (12, 28,28)
        self.pool = nn.MaxPool2d(2,2) # (12, 14, 14)

        self.conv2 = nn.Conv2d(12, 24, 5) # (24,10,10)
        self.fc1 = nn.Linear(24*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = NeuralNet()
net.load_state_dict(torch.load('trained_img_Classifier.pth', weights_only=True))
net.eval()

with torch.no_grad():
    for image in images:
        output = net(image)
        _, predicted = torch.max(output, 1)
        print(f'Prediction: {class_names[predicted.item()]}')