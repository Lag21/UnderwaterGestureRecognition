import timm
import torch
import numpy as np
import mediapipe as mp
from collections import OrderedDict
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from spatialAttentionModule import GestureRecognition as GR

def GestureRecognition():
    model = LoadModel(pathfile='trainedModels/inception-v3-tlearning02.pth')
    model.eval()

    imageTensor = LoadImage('test_images/peace.jpg')
    #imageTensor = imageTensor.movedim(0,2)
    #plt.imshow(imageTensor.movedim(0,2))
    #plt.show()
    #print(imageTensor)
    inputBatch = imageTensor.unsqueeze(0)
    with torch.no_grad():
        output = model(inputBatch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    estClass = np.argmax(probabilities)
    print(estClass)


def LoadModel(pathfile, cleanStateDict=True):
    model = timm.create_model('inception_v3',pretrained=False, num_classes=27)
    if cleanStateDict:
        state_dict = torch.load(pathfile)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[6:]
            new_state_dict[name] = v 
    
    model.load_state_dict(new_state_dict)
    return model

def LoadImage(pathfile):
    image = Image.open(pathfile)
    trans = transforms.Compose([
        transforms.Resize(299),
        #transforms.CenterCrop(299),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imageTensor = trans(image)
    return imageTensor

if __name__=='__main__':

    GR(device='cuda')
    #GestureRecognition()