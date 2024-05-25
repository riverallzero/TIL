# SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image

## Structures
![](../asset/segmentation/segnet-structure.png)

### encoder
Consists of **13 convolutional layers**, identical to the **VGG16** network's initial layers designed for object classification.
Each encoder layer convolves with a filter bank, applies batch normalization, ReLU activation, and max-pooling with a 2x2 window and stride 2 for sub-sampling.
To maintain boundary information crucial for segmentation, SegNet stores **max-pooling indices** instead of feature maps, enhancing efficiency in memory usage. 

#### max-pooling indices
<img src="../asset/segmentation/segnet-max-pool-indices.png" width="40%"/>

this is the method that refers to the positions of the maximum elements within each pooling window during the max-pooling operation.

### decoder
Complements the encoder by mapping low-resolution encoder feature maps to full input resolution for **pixel-wise** classification.
It uses pooling indices from the encoder to perform **non-linear upsampling**, eliminating the need to learn upsampling and enhancing boundary delineation.
The final decoder output is fed to a soft-max classifier for pixel-wise classification, making the architecture fully convolutional without fully connected layers. 

## Code
```python

```
