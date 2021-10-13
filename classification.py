import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from PIL import Image
from torch.autograd import Variable


data_transforms = {
    "train" : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Resize((256,256))
    ])
,
    "test" : transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
}
class_names = ['Chile Limon', 'classicsalted', 'cream_and_onion', 'hot_and_sweet', 'indian_magic_masala', 'max_chili', 'maxx_sizzling_barbeque', 'tangy_tomato']

def load_model():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft.load_state_dict(torch.load("models/resnet18_classifier" , map_location=torch.device('cpu')))
    model_ft.eval()
    return model_ft

def predict_image(image, model):
    image_tensor = data_transforms["test"](image).float()
    #image_tensor = image.unsqueeze_(0)
    image_tensor = image_tensor.unsqueeze_(0)
    input=Variable(image_tensor)

    output = model(input)
    #print(output)
    #print('max:',max(output[0]))
    index = output.data.numpy().argmax()
    pred_flv = class_names[index]
    pred_flv_score = np.round((max(output[0]).item())/10, decimals=3)
    return pred_flv,pred_flv_score


if __name__ == "__main__":
        model = load_model()
        #print(model)
        root = "data/for_classification/"
        imgList = os.listdir(root)
        #print(imgList)
        for imgName in imgList:
            path = root + imgName 
            img = Image.open(path) 
            print(imgName ," : ", predict_image(img, model))
