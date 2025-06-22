# Coin Detection Project

## About The Project

This project implements a comprehensive computer vision pipeline for detecting and classifying coins in digital images. The system uses advanced image processing techniques to automatically identify, locate, and classify different coin denominations. Two versions are provided: a base detection system and an enhanced version with coin classification capabilities.

The pipeline processes images through multiple stages including greyscale conversion, edge detection, noise reduction, morphological operations, and connected component analysis to accurately detect circular coin objects and draw bounding boxes around them.

## Features

### Base Version
- **Coin Detection**: Automatically detects circular coin objects in images
- **Edge Detection**: Uses Scharr filters for robust edge detection
- **Morphological Processing**: Applies dilation and erosion for noise reduction
- **Bounding Box Generation**: Creates precise rectangular boundaries around detected coins
- **Size Filtering**: Filters out small artifacts using minimum area thresholds

### Extension Version
- **All Base Features**: Includes everything from the base version
- **Coin Classification**: Identifies specific coin denominations (10¢, 20¢, 50¢, $1, $2)
- **Circularity Validation**: Validates detected objects are circular using area-to-radius ratios
- **Enhanced Edge Detection**: Uses Laplacian filters for improved edge detection
- **Visual Annotations**: Displays coin types and total count on output images
- **Improved Processing Order**: Optimized erosion-then-dilation sequence for better results

## Technical Implementation

### Image Processing Pipeline

#### 1. Preprocessing
- **RGB to Greyscale Conversion**: Uses weighted formula (0.3R + 0.6G + 0.1B)
- **Contrast Enhancement**: Percentile-based mapping for improved contrast (base version only)

#### 2. Edge Detection
- **Base Version**: Scharr horizontal and vertical edge filters with magnitude calculation
- **Extension Version**: Laplacian filter for enhanced edge detection with absolute value conversion

#### 3. Noise Reduction
- **Mean Blurring**: 5×5 kernel applied multiple times for smooth results
- **Adaptive Iterations**: 3 iterations (base) / 2 iterations (extension) for optimal smoothing

#### 4. Segmentation
- **Binary Thresholding**: Converts processed images to binary format
- **Threshold Values**: 22 (base) / 20 (extension) for optimal separation

#### 5. Morphological Operations
- **Kernel Design**: Custom 5×5 circular kernel for coin-like shapes
- **Base Version**: 4 dilations followed by 4 erosions
- **Extension Version**: 4 erosions followed by 4 dilations (improved sequence)

#### 6. Object Analysis
- **Connected Component Labeling**: BFS algorithm for region identification
- **Size Filtering**: Minimum 10,000 pixels to eliminate small artifacts
- **Circularity Check**: Area validation against expected circular area (extension only)

#### 7. Classification (Extension Only)
Coin classification based on pixel area:
- **10 Cent**: < 39,000 pixels
- **20 Cent**: 39,000-41,000 pixels  
- **1 Dollar**: 41,000-45,000 pixels
- **50 Cent**: 45,000-54,000 pixels
- **2 Dollar**: 54,000-100,000 pixels
- **Unknown**: > 100,000 pixels

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Required libraries: `matplotlib`

### Installation Steps

1. **Clone or download the project files**

2. **Install required dependencies**:
   ```bash
   pip install matplotlib
   ```

3. **Verify directory structure**:
   Ensure the `imageIO` folder and PNG reader library are present

## Usage

### Base Version
```bash
python coin_detection_base.py
```

### Extension Version  
```bash
python coin_detection_extended.py
```
