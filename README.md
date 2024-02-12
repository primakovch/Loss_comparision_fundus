# Loss_comparision_fundus

This project aims to study the effect of various loss functions on the performance of unet model to perform semantic segmentation on DRIVE dataset. The drive dataset[1] is a medical dataset consisting of fundus images. The target images are binary masks with white pixel values for nerve regions. The solution pipeline is as follows:
1. Reading the image files and preliminary preprocessing (image_extraction.ipynb)
2. Image augmentation to increase the training samples (augmentation.ipynb) and merging them into a single numpy file (file_merger.ipynb)
3. Training the model iteratively, for each loss function (Pytorhc1/train.ipynb)
4. Result analysis and visulization. (performances record/loss_analysis.ipynb)

The result of this study is published as a conference paper, titled "Study of Loss Functions on Retinal Vessel Segmentation using UNET Architecture" at IEEE Xplore. The paper is availble at: https://ieeexplore.ieee.org/document/9688289
