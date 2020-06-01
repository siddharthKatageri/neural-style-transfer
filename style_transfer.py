import torch
import numpy
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# set working device
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# defining function for getting the activations of our decided layers




def img_load(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).to(device)
    img  = img.unsqueeze(0)
    return img

def img_convert_to_show(img):
    x = img.cpu().clone().numpy()
    x = x.squeeze(0)
    x = x.transpose(1,2,0)
    x = x * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    #x = x * (0.5,0.5,0.5) + (0.5,0.5,0.5)
    return x



transform = transforms.Compose([transforms.Resize(300),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                ])


content = img_load("./content.jpg")
style = img_load("./style.jpg")
print(content.shape)
print(style.shape)


fig, (ax1,ax2) = plt.subplots(1,2)

ax1.imshow(img_convert_to_show(content),label = "Content")
ax2.imshow(img_convert_to_show(style),label = "Style")
plt.show()





'''
# download vgg19 pretrained model
model = models.vgg19(pretrained=True)
model = model.features

layers = {
    '0' : 'conv1_1',
    '5' : 'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '21': 'conv4_2',
    '28': 'conv5_1'
    }
'''
