# 2nd place solution ： Segmentation + 2.5D CNN + GRU Attention

Thanks to Kaggle and RSNA for such a great competition, we are very happy to have finished second. From this complex game, we tried to find the most concise and efficient solution, and gained a lot of knowledge. It was also one of the most hard-drive intensive competition I've ever seen, and we wasted time loading data because we didn't have enough space to save the high-resolution pseudo-label voxel.

To cut to the chase, our solution also consists of two stages, and use 2.5D CNNs, which we learned from [@Awsaf](https://www.kaggle.com/awsaf49) in UWM [UWMGI: 2.5D [Train\] [PyTorch] | Kaggle](https://www .kaggle.com/code/awsaf49/uwmgi-2-5d-train-pytorch) Harvesting a lot.

stage1： 2.5D CNN + Unet for Segmentation

stage2： CNN + BiGRU + Attention for Classification



## Stage 1

First, we used the 87 Study Segmentation samples provided by the organizers . We recreated the mask labels according to the following way

```
0 ---> background
1 ---> C1
2 ---> C2
...
8 ---> T1 - T12
```

We used the more general 2.5D and get 3 channels of image data, i.e., the original image and its sides: i-1, i, i+1, with each sample of size 320 * 320 * 3. stride=1, which means that each slice will be read 3 times, which looks very bulky, but the number of samples is small and training is still relatively fast. Combined with our use of higher resolution segmentation maps, we can achieve a dice score of 0.96 in about 10 epochs.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3799319%2F29389b33392e4b84d56d0d53be39c23e%2FSnipaste_2022-11-09_01-18-47.png?generation=1668028486937787&alt=media)




The data augmentation section here is as follows, similar to [UWMGI: 2.5D [Train\] [PyTorch] | Kaggle](https://www.kaggle.com/code/awsaf49/uwmgi-2-5d-train-pytorch), without much change.

We also tried heavier data augmentation, but it did not work better.

```
Resize(CFG.img_size, CFG.img_size, interpolation=cv2.INTER_NEAREST),
HorizontalFlip(p=0.5),
ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
OneOf([
    GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
    ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
], p=0.25),
```



For segmentation model，we used segmentation_models_pytorch lib， backbone was efficientnet-b0，decoder was unet。

optimizer="AdamW"

scheduler="CosineAnnealingLR" + "GradualWarmupSchedulerV3"





## Crop Voxel

Once we trained the segmentation model, we generalized it to all 2019 study, we did the same way as before for the input data, and after the model predicted the results, we manually looked at several predicted images and found that the accuracy was pretty good.

We croped out all 7 cervical vertebrae of each study separately, each cervical to a fracture label (need to use train.csv corresponding on), on each cervical, there are 24 channels (slice), According to our EDA, most of the studies contain 200-300 slices, so the average to each cervical is about 30 slices. we choose 24 slices, which will be satisfied by most cervicals. For cervicals larger than 24 slice, we use a simple numpy function to get 24 slice evenly.

```
sample_index = np.linspace(0, len(one_study_cid)-1, sample_num, dtype=int)
```

One of the challenges for us was the training images for this competition are 300GB, and if we were to save the cropped 3D high-resolution training images locally, it would exceed the capacity of the hard disk, so we are forced to choose to record the cropped, [x0:x1, y0:y1, z0:z1] and the corresponding slice's dcm file number for the training process in stage2 for reading and cropping.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3799319%2F4ab026464572fb40996597d4aaa424c2%2Fimage-20221110043219911.png?generation=1668028409059020&alt=media)



## Stage 2



After we crop, the data shape of our input was (bs, 24, img_size, img_size), 24 channels, representing 24 uniformly distributed slices, and also the seq_len of GRU.

For data sampling, we ignored the wrong study 1.2.826.0.1.3680043.20574 and 1.2.826.0.1.3680043.29952

Regarding data augmentation, we used methods similar to stage1 with a little new augmentation.

For the model we used CNN + biGRU + Attention, where for the CNN backbone we used tf_efficientnetv2_s and resnest50d from the timm library. For some other details, we initialized the GRU, since it seems that the original GRU weights on Pytorch are not very good. We also added SpatialDropout , which also gives us a little improvement.



## Things we didn't had time to do

1. use bbox csv in yolo
2. Transformer for sequential model
3. buy new hard-drive :)





