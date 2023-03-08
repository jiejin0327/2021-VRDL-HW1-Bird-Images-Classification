# 2021 VRDL HW1 Bird Images Classification

This project is bird species classification using deep learning models. 

The dataset contained 6,033 bird images belonging to 200 bird species, e.g., tree sparrow or mockingbird (training: 3,000, test: 3,033)

## Introduction
This project is a bird classifier that uses the PyTorch framework and the ResNet50 model. It leverages data augmentation, learning rate adjustment and other techniques to enhance the accuracy. It can recognize the species of birds in images based on their visual features. It supports 200 different categories of birds. 

## Requirements
To run this project, you need to install the following packages:

- torch
- torchvision
- numpy
- pandas
- sklearn

## Usage
To product my submission without retraining, do the following step:

1. Download [data](https://competitions.codalab.org/my/datasets/download/83f7141a-641e-4e32-8d0c-42b482457836)
  
   We need these data to get submission:
  
    - classes.txt
    - testing_images(test image putting in this Folder)
    - testing_img_order.txt  
    
2. Download [Pretrained model](https://drive.google.com/uc?export=download&id=1yKz2pEB2N6u9DKrmtio9-RaDM3h29u6s)
3. Download [Inference.py(make submission)](https://drive.google.com/uc?export=download&id=1MxxValX4DfHhJn0c8A4CPdWX6Vo7S87R)

   Please put (1)(2)(3)data in the same folder

4. Run the following command to start the submission:
```
!python inference.py --model_path model.pth --data_path classes.txt --test_dir testing_images --test_label testing_img_order.txt
```
## Results
The experiment results show that the pre-trained model can achieve higher performance even on a different dataset. In this assignment, I applied the pre-trained model ResNet50 and used data augmentation to enrich the training data. Moreover, I adopted the author’s padding strategy to preprocess the data into a suitable size for enhancing the model performance. With this padding step, my scores improved from 64 to 67. After several days of model tuning, I obtained **0.676888** scores on codalab finally.

## Reference
[caltech-birds-advanced-classification](https://github.com/slipnitskaya/caltech-birds-advanced-classification)
