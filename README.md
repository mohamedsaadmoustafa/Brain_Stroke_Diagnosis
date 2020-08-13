# Brain Stroke Segmentation


## Dataset
  • ATLAS ‘Anatomical Tracings of Lesions After Stroke’, 
  • an open-source dataset of 229 T1-weighted MRI scans (n=220) with manually segmented lesions and metadata. 
    This large, diverse dataset can be used to train and test lesion segmentation algorithms 
    and provides a standardized dataset for comparing the performance of different segmentation methods.

## The dataset includes:
    • 229 T1-weighted MRI scans (n=220) with lesion segmentation
    • MNI152 standard-space T1-weighted average structural template image
    • A .csv file containing lesion metadata

## Data Preparation:
   • Each 3D volume in the dataset has a shape of ( 197, 233, 189 ). 
   •  Each deface “MRI” has a ground truth consisting of at least one or more masks. 
      Multiple masks problem solved with Combine Lesions function using logical OR 
      to create a single mask for every deface “MRI” that contains all lesions from each mask.
    • Apply diffrent image preprocessing methods.
    • Data Augmentation

## Models 'trails':
    • BCDC-Net 
    • X-Net
  
## Evaluation metrics: 
    • Dice's coefficient
    • Intersection Over Union (iou)
    • pixel accuracy
    
## Loss function: 
    • dice loss

Optimization function: 
    • Adam

## Learning Rate scheduler: 
    Learning rate schedules seek to adjust the learning rate during training 
    by reducing the learning rate according to a predefined schedule. 
    Common learning rate schedules include: 

        • time-based decay: r = lr *  ( 1. / ( 1. + self.decay * self.iterations ) )

        • step decay: lr = lr0 * drop ^ floor( epoch / epochs_drop )

        • exponential decay: lr = lr0 * e ^ ( −kt )
    
## Post Processing:
    • Predicted masks have uncertainty values that show as a salt and pepper noise in many predicted masks 
      so we use a threshold at 0.5 to clear those uncertainty values from our predicted mask.
    • Second, To increase segmented areas we use dilation with a kernel of 5x5 of ones.

 
## Future Work:
    Insead of 2D models try 3D models like 3D residual convolutional neural network.
