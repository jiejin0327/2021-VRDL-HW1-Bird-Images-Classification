# bird-species-classification

To product my submission without retraining, do the following step:
  1. [Download data](https://competitions.codalab.org/my/datasets/download/83f7141a-641e-4e32-8d0c-42b482457836)
  We need these data to get submission:
    * classes.txt
    * testing_images(test image putting in this Folder)
    * testing_img_order.txt  
    
    * 昵称：果冻虾仁
    - 别名：隔壁老王
    * 英文名：Jelly

  
  2. [Download Pretrained model](https://drive.google.com/uc?export=download&id=1yKz2pEB2N6u9DKrmtio9-RaDM3h29u6s)
  3. [Download Inference.py(make submission)](https://drive.google.com/uc?export=download&id=1MxxValX4DfHhJn0c8A4CPdWX6Vo7S87R)

Please put (1)(2)(3)data in the same folder

  4. Run the following command to start the submission: 
```
!python inference.py --model_path model.pth --data_path classes.txt --test_dir testing_images --test_label testing_img_order.txt
```
