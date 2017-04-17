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

## Upcoming Changes/Enhancements
PyMedImage will be updated periodically when time permits to become more functional and robust. Please stay tuned.
1. Unit Tests
2. Documentation Page
3. Incorporation of more specific research oriented operations that can be chained together into a processing pipeline.
4. Examples and Getting Started Guide

## Contributing
If you'd like to get involved in contributing to this project, contact Ryan Neph at neph320@gmail.com.
