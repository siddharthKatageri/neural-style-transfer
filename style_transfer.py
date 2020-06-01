import torch
import numpy
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# set working device
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#load image and apply transfromations
def img_load(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).to(device)
    img  = img.unsqueeze(0)     # converts image shape from (c, w, h)->(1, c, w, h)| Required for model input
    return img

#function for bringing back the transformed image to original shape and intensity
def img_convert_to_show(img):
    x = img.cpu().clone().numpy()
    x = x.squeeze(0)            #converts image shape from (1, c, w, h)->(c, w, h)
    x = x.transpose(1,2,0)      # converts image shape from (c, w, h)->(w, h, c)
    x = x * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    #x = x * (0.5,0.5,0.5) + (0.5,0.5,0.5)
    return x

# defining function for getting the activations of our decided layers
def get_activations_from_model(input, model):
    layers_style = {
    '0' : 'conv1_1',
    '5' : 'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '28': 'conv5_1'
    }

    layers_content = {
    '21': 'conv4_2'
    }

    features_content = {}
    features_sytle = {}
    x = input
    for name, layer in model._modules.items():
        x = layer(x)
        if(name in layers_style):
            features_sytle[layers_style[name]] = x
        if(name in layers_content):
            print(layer)
            features_content[layers_content[name]] = x

    return features_content, features_sytle





# download vgg19 pretrained model
model = models.vgg19(pretrained=True)
model = model.features

#defined transforms
transform = transforms.Compose([transforms.Resize(300),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                ])


#load images
content = img_load("./content.jpg")
style = img_load("./style.jpg")
print(content.shape)
print(style.shape)


#image show
fig, (ax1,ax2) = plt.subplots(1,2)

ax1.imshow(img_convert_to_show(content),label = "Content")
ax2.imshow(img_convert_to_show(style),label = "Style")
#plt.show()


#get activations for content loss and style loss of both content and style images
cimg_activation_for_content_loss, cimg_activation_for_style_loss = get_activations_from_model(content, model)
simg_activation_for_content_loss, simg_activation_for_style_loss = get_activations_from_model(style, model)

print(cimg_activation_for_style_loss['conv1_1'].shape)
print(simg_activation_for_style_loss['conv1_1'].shape)

