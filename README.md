## \*\*PROJECT RELOCATION\*\*
I have moved the most popular functionality of this library (namely the image load/save, and RTStruct mask load/mask-generation functions) to a more actively maintained repository: [RTTypes](https://github.com/ryanneph/rttypes)

For continued availability of the feature calculation framework, you may continue to use this library, though maintainence will be less frequent.

---

# PyMedImage
This is a library written in Python 3.x and CUDA for common tasks when dealing with the DICOM medical image format in a research setting.

## Overview
Key features include:
* Easy I/O of dicom images/volumes commonly used in storage of CT, MR, and PET images.
* Reading and conversion of Radiotherapy contours/masks from .rtstruct file to 2D/3D binary masks.
* Patch-based (local) image feature calculation including:
  * 1st Order Statistics
  * Gray Level Co-Occurence Matrices (GLCM)
  * Gray Level Run-Length Matrices (GLRLM)
  * Wavelet Decomposition
  * Customizable Haar-Like features
* GPU-Acceleration and Multi-process management
* Customizable Logging Utilities
* Image-feature-based voxel clustering

## Installing
Open a terminal window and enter:
``` bash
pip3 install git+git://github.com/ryanneph/PyMedImage.git#egg=PyMedImage
```

## Updating
Open a terminal window and enter:
``` bash
pip3 install --upgrade git+git://github.com/ryanneph/PyMedImage.git#egg=PyMedImage
```

## Development
Open a terminal window and enter:
``` bash
git clone https://github.com/ryanneph/PyMedImage.git
cd PyMedImage
pip3 install -e .
```
