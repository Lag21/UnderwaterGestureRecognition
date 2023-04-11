import timm
import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
from torch import optim, nn, utils, Tensor
from torchvision import transforms


class DepthEstimator(nn.Module):
  def __init__(self, device="cpu"):
    super().__init__()
    self.device = device
    self.depthEstimator = nn.Sequential(nn.Linear(261, 250),
                               nn.Linear(250, 250),
                               nn.Linear(250, 150),
                               nn.Linear(150, 150),
                               nn.Linear(150, 100),
                               nn.Linear(100, 100),
                               nn.Linear(100, 75),
                               nn.Linear(75, 50),
                               nn.Linear(50, 3)
                               )

  def forward(self, x):
    x = torch.tensor(x, dtype=torch.float32)
    return self.depthEstimator(x)
  
class StaticGestureRecognizer():
  def __init__(self, device="cpu"):
     super().__init__()
     self.device = device
     self.inceptionTransform = transforms.Compose([
        transforms.Resize(299),
        #transforms.CenterCrop(299),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

  def LoadModel(self, pathfile, cleanStateDict=True):
    self.model = timm.create_model('inception_v3',pretrained=False, num_classes=27)
    if cleanStateDict:
        state_dict = torch.load(pathfile)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[6:]
            new_state_dict[name] = v 

    self.model.load_state_dict(new_state_dict)

    self.activation = {}
    def get_activation(name):
      def hook(model, input, output):
        self.activation[name] = output.detach()
      return hook
    
    self.model.global_pool.register_forward_hook(get_activation('avgpool'))
    self.model.eval()
    #return model

  def LoadImage(self, pathfile):
    image = Image.open(pathfile)
    self.imageTensor = self.inceptionTransform(image)
    #return imageTensor
  
  def LoadArray(self, imageArray):
    image = Image.fromarray(imageArray)
    self.imageTensor = self.inceptionTransform(image)

  def forward(self):
    inputBatch = self.imageTensor.unsqueeze(0)
    with torch.no_grad():
      output = self.model(inputBatch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    estClass = np.argmax(probabilities)
    maxProbability = torch.max(probabilities)
    print(f'Recognizing gesture {estClass} with a confidence of {maxProbability}')

  def GetEmbedding(self):
    inputBatch = self.imageTensor.unsqueeze(0)
    with torch.no_grad():
      self.model(inputBatch)
    return self.activation['avgpool']
