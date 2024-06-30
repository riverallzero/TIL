# CNN: Convolutional Neural Network

Let's start by understanding how a Convolutional GNN works with ```X``` and ```O``` classifier.

![](../asset/neural-network/cnn-x.png)

First, let's consider this image and determine how we can classify it as an X. 
CNNs compare images ```piece by piece```, extracting features that represent the image well.

In this image, the 3x3 feature can be represented as follows:

<img src="../asset/neural-network/cnn-feature.png" width="60%">

This is the important part that shows this image is an ```X```.

## Convolution

Using this feature as a filter, we convolve it with the image, ```calculating the average``` and placing it at the center.

![](../asset/neural-network/cnn-convolution.png)

If we calculate the feature map for the entire features, the resulting image will look like this:

<img src="../asset/neural-network/cnn-convolution-result.png" width="60%">

## Pooling
Pooling methods like max-pooling, average-pooling, min-pooling, and sum-pooling are used to condense the feature. Here, i use ```max-pooling```.
In a **2x2 grid**, we select the largest value. This pooling operation is applied to all the feature maps.

<img src="../asset/neural-network/cnn-pooling.png" width="60%">

## ReLU(Rectified Linear Units)
ReLU is activation function wherever a negative number occurs, it is ```swapped out for a 0```. 
This helps the CNN stay **mathematically stable** by preventing learned values from getting stuck near 0 or blowing up toward infinity.

<img src="../asset/neural-network/cnn-relu.png" width="60%">

## Stacking
Layers can be repeated several (or many) times. In this case, let's stack ```conv > relu > conv > relu > max-pool > conv> relu > max-pool```

![](../asset/neural-network/cnn-stack.png)

## Fully Connected Layer

<img src="../asset/neural-network/cnn-fcn.png" width="60%">

Instead of treating inputs as a two-dimensional array, they are treated as a single list, and all values are treated identically. 
Every value gets its ```own vote``` on whether the current image is an X or O to classify.
