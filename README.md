# Fast ComICS

Fast Comprehensive Image Clustering by Similarity (Fast ComICS) is a method
to determine which images are similar to one another and combining them into
a cluster.

```
Similarity (confusion) matrix := ![ A*A' + !(A) * !(A') ]

This alone however is not yet enough... If vecA is all 1 and vecB is all 0 the result of vecA x vecB is 0

Use sum of digits some how... Maybe...

Possibly using larger of either sum of digits of vector as expected similarity value
```

# Requirements
 - Python
   - [ImageHash](https://github.com/JohannesBuchner/imagehash)
   - [NumPy](https://numpy.org/)
   - [matplotlib](https://matplotlib.org/)


# Dataset
The dataset provides collections of two hash lengths (64bit, 1024bit) containing the following:

 - The original image's hashes
 - Randomized augmentation of original
 - Noisy augmentation of original

The pickle file contains following forensic image hashes:

 - average
 - perceptual
 - difference
 - wavelet
 - color
 - crop-resistant

Naming scheme of pickle files:

```
<image-id>-<variant: {o,n,r}>-<variation-id>.pkl
```

 - `image-id`: id of the original image
 - `variant`:
   - `o`: original
   - `n`: noisy
     - salt and peper: `amount=0.02`
     - gaussian noise: `mean=0.15`
     - random rotation: `angle=[-25, 25]/10 deg`
   - `r`: randomized
     - added noise (see `n`)
     - 10 added random words at random location
 - `variant-id`: id of repeated variant


# Quick Start
