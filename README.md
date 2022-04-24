# MMDReg
Deformable registration for histopathology images using multi-magnification structure

## About
We present an end-to-end unsupervised deformable registration approach for high-resolution histopathology images with different stains. Our method comprises two sequential registration networks, where the local affine network can handle small deformations, and the non-rigid network is able to align texture details further. Both networks adopt the multi-magnification structure to improve registration accuracy. A dataset containing 41 multistained histopathology whole-slide-images provided by the Frankfurt University Hospital was pre-aligned and used as the training and validation set for the preposed method.
![Image text](https://github.com/CMYXing/MMDRegistration/blob/main/img_folder/algorithm.png)
![Image text](https://github.com/CMYXing/MMDRegistration/blob/main/img_folder/network architectures.png)

## Technical Details
The proposed algorithm was implemented by modifying and extending the DeepHistReg framework. https://github.com/MWod/DeepHistReg

## Acknowledgmnets
### Paper (to be published)
https://openreview.net/forum?id=BAMUUQR25CK
### Other references
Marek Wodzinski and Henning Müller, DeepHistReg: Unsupervised Deep Learning Registration Framework for Differently Stained Histology Samples. Computer Methods and Programs in Biomedicine Vol. 198, January 2021. 
https://www.sciencedirect.com/science/article/pii/S0169260720316321 
The article presents the whole DeepHistReg framework with deep segmentation, initial alignment, affine registration and improved deformable registration.
Marek Wodzinski and Henning Müller, Unsupervised Learning-based Nonrigid Registration of High Resolution Histology Images, 11th International Workshop on Machine Learning in Medical Imaging (MICCAI-MLMI), 2020.
https://link.springer.com/chapter/10.1007/978-3-030-59861-7_49 
The article introduces the first version of the nonrigid registration.

