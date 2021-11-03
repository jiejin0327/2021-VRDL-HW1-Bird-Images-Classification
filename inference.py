# -*- coding: utf-8 -*-

import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm

from PIL import Image
from argparse import ArgumentParser

"""
Load data
"""

class MyDataset(Dataset):
  def __init__(self , img_dir, label_path, transform = None):

    label_data = open(label_path, 'r')
    imgs = []

    for x in label_data:
      x = x.rstrip()
      imgs.append(x)
        
      self.img_path = [os.path.join(img_dir,x) for x in imgs]
      self.label = [int(x[0].split(".")[0]) for x in imgs]

    self.transform = transform

  def __getitem__(self,index):
    img_path = self.img_path[index]
    label = self.label[index]
    img = Image.open(img_path).convert("RGB")
    
    if self.transform is not None:
      img = self.transform(img)
    return img,label

  def __len__(self):
    return len(self.img_path)

def pad(img, fill=0, size_max=500):
    """
    Pads images to the specified size (height x width). 
    Fills up the padded area with value(s) passed to the `fill` parameter. 
    """
    pad_height = max(0, size_max - img.height)
    pad_width = max(0, size_max - img.width)
    
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)


def get_test_transform():
  fill = tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406)))
  max_padding = transforms.Lambda(lambda x: pad(x, fill=fill))

  test_transform = transforms.Compose([max_padding,
                      transforms.CenterCrop((375, 375)),                     
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])])
  return test_transform

def get_test_datasets(test_dir,test_label):
  test_transform = get_test_transform()

  test_datasets = MyDataset(img_dir = test_dir,
                label_path = test_label,
                transform = test_transform)
  return test_datasets

def get_class_data(data_path):
  with open(data_path) as f:
    class_data = [ x.strip() for x in f.readlines()]
  return class_data

def get_test_loader(test_datasets):
  batch_size = 24
  test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=2)
  return test_loader

"""
Load Pretrain Model
"""

def load_model(model_name,class_num):
  if os.path.isfile(model_name):
    model = models.resnet50(pretrained = False)
    fc_features = model.fc.in_features 
    model.fc = nn.Linear(fc_features, class_num)
    model.load_state_dict(torch.load(model_name))

    device = torch.device("cuda")
    if torch.cuda.is_available():
      model = model.to(device)
      
  else :
    print("【Error】: The pre-train model is NOT in file.")

  print("Load pre-trained model success")
  return model

"""
submission

"""

def submission(model, datasets, loader, class_data):
  print("Start to submission")
  model.eval()
  batch_size = 24
  with torch.no_grad():
    submission = []
    index = 0
    for data in tqdm(loader):
      img, label = data
      img = img.cuda()

      output = model(img) # the predicted category 
      predicted_class=torch.max(output,1).indices

      for i in range(len(predicted_class)):
        img_path = datasets.img_path[index * batch_size + i].split("/")[1]
        submission.append([img_path, str(class_data[int(predicted_class[i])])])

      index += 1

  np.savetxt('answer.txt', submission, fmt='%s')

def run_pretrained_model(args):
  class_data = get_class_data(args.data_path)
  class_num = len(class_data)
  model = load_model(args.model_path, class_num)
  test_datasets = get_test_datasets(args.test_dir, args.test_label)
  test_loader = get_test_loader(test_datasets)
  submission(model, test_datasets, test_loader, class_data)

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--model_path', default='resnet10.pth', help="Model Path")
  parser.add_argument('--data_path', default='classes.txt', help="Class Data Path")
  parser.add_argument('--test_dir', default='testing_images', help="Testing Image Path")
  parser.add_argument('--test_label', default='testing_img_order.txt', help="Testing Label Path")
  args = parser.parse_args()
  run_pretrained_model(args)