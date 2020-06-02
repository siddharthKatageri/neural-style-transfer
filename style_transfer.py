import torch
import numpy as np
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
    x = img.cpu().clone().detach().numpy()
    x = x.squeeze(0)            #converts image shape from (1, c, w, h)->(c, w, h)
    x = x.transpose(1,2,0)      # converts image shape from (c, w, h)->(w, h, c)
    x = x * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    #x = x * (0.5,0.5,0.5) + (0.5,0.5,0.5)
    return x

# defining function for getting the activations of our decided layers
def get_activations_from_model(input, model, label):
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

    if(label == "content"):
        for name, layer in model._modules.items():
            x = layer(x)
            if(name == '21'):
                features_content[layers_content[name]] = x
        return features_content

    if(label == "style"):
        for name, layer in model._modules.items():
            x = layer(x)
            if(name in layers_style):
                features_sytle[layers_style[name]] = x
        return features_sytle

    if(label == "target"):
        for name, layer in model._modules.items():
            x = layer(x)
            if(name in layers_style):
                features_sytle[layers_style[name]] = x
            if(name in layers_content):
                features_content[layers_content[name]] = x
        return features_content, features_sytle

def gram_matrix(G):
    m, c, w, h = G.shape
    G = G.view(m*c, w*h)
    gram_mat = torch.mm(G,G.t())
    return gram_mat

def compute_content_loss(c_activation, t_activation):
    m, c, w, h = c_activation.shape
    c_activation = c_activation.view(m*c, w*h)
    t_activation = t_activation.view(m*c, w*h)
    content_loss = torch.sum((c_activation - t_activation)**2)/(4*h*w*c)
    return content_loss

def compute_style_loss(s_gram, t_activation):
    m, c, w, h = t_activation.shape
    t_gram = gram_matrix(t_activation)
    style_loss = torch.sum((s_gram - t_gram)**2)/(4*c*c*(w*h)**2)
    return style_loss







# download vgg19 pretrained model
model = models.vgg19(pretrained=True)
model = model.features
for p in model.parameters():
    p.requires_grad = False
model.to(device)

#defined transforms
transform = transforms.Compose([transforms.Resize(300),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                ])


#load images
content = img_load("./content.jpg")
style = img_load("./style.jpg")



#image show
fig, (ax1,ax2) = plt.subplots(1,2)

ax1.imshow(img_convert_to_show(content),label = "Content")
ax2.imshow(img_convert_to_show(style),label = "Style")
#plt.show()


#get activations for content image and style image
content_image_activation = get_activations_from_model(content, model, "content")
style_image_activations = get_activations_from_model(style, model, "style")


# clone the content image to create a target image
target = content.clone().requires_grad_(True).to(device)


#gram matrix for all the layers for style loss
grams_for_style = {layer:gram_matrix(style_image_activations[layer]) for layer in style_image_activations}

'''
style_weights = {
                "conv1_1" : 0.2,
                 "conv2_1" : 0.2,
                 "conv3_1" : 0.2,
                 "conv4_1" : 0.2,
                 "conv5_1" : 0.2
                }
'''
style_weights = {
                "conv1_1" : 0.8,
                 "conv2_1" : 0.1,
                 "conv3_1" : 0.4,
                 "conv4_1" : 0.2,
                 "conv5_1" : 0.1
                }

alpha = 100
beta = 1e8

epochs = 1000

optimizer = torch.optim.Adam([target], lr=0.1)

for i in range(1, epochs+1):
    target_image_activation_for_content, target_image_activation_for_style = get_activations_from_model(target, model, "target")
    c_loss = compute_content_loss(content_image_activation['conv4_2'], target_image_activation_for_content['conv4_2'])

    s_loss = 0
    for layer in style_weights:
        take = compute_style_loss(grams_for_style[layer], target_image_activation_for_style[layer])
        s_loss += take

    total_loss = alpha*c_loss + beta*s_loss

    optimizer.zero_grad()
    total_loss.backward(retain_graph=True)
    optimizer.step()

    if i == 0:
        print("Started!\n")

    if i%10 == 0:
        print("Epoch:", i, ":", total_loss)
    if i%500 == 0:
        plt.imsave('./output/1/'+str(i)+'.png',img_convert_to_show(target),format='png')
'''
# get activations for target image



compute_content_loss(content_image_activation['conv4_2'], target_image_activation_for_content['conv4_2'])
'''
