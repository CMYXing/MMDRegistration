# MMDReg
Deformable registration for histopathology images using multi-magnification structure

## About
We present an end-to-end unsupervised deformable registration approach for high-resolution histopathology images with different stains. Our method comprises two sequential registration networks, where the local affine network can handle small deformations, and the non-rigid network is able to align texture details further. Both networks adopt the multi-magnification structure to improve registration accuracy. A dataset containing 41 multistained histopathology whole-slide-images provided by the Frankfurt University Hospital was pre-aligned and used as the training and validation set for the preposed method.
