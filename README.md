# Coin Detection Pipeline

## About The Project

This project involves developing a Python pipeline to detect and outline coins in images using computer vision techniques. The pipeline implements various image processing methods including greyscale conversion, edge detection, blurring, thresholding, morphological operations, connected component analysis, and bounding box detection.

The system is designed to automatically identify and locate coins in digital images, making it useful for applications such as automated counting systems or inventory management.

## Project Steps

### 1. Convert to Greyscale and Normalize
- **Convert to Greyscale**: Transform RGB images to greyscale using the standard luminance formula (0.3 × Red + 0.6 × Green + 0.1 × Blue)
- **Contrast Stretching**: Normalize pixel values to the full 0-255 range using percentile-based stretching

### 2. Edge Detection
- **Scharr Filter**: Apply 3×3 Scharr filters in both horizontal and vertical directions
- **Edge Strength**: Calculate edge magnitude by combining horizontal and vertical edge responses

### 3. Image Blurring
- **Mean Filter**: Apply 5×5 mean filtering to reduce noise and smooth the image
- **Multiple Passes**: Perform filtering operations sequentially to enhance results

### 4. Threshold the Image
- **Binary Segmentation**: Convert the processed image to binary format using optimal threshold values
- **Foreground/Background Separation**: Isolate coin regions from the background

### 5. Morphological Operations
- **Erosion and Dilation**: Apply morphological operations using circular 5×5 kernels
- **Noise Reduction**: Remove small artifacts and fill gaps in detected regions

### 6. Connected Component Analysis
- **Component Detection**: Identify separate connected regions in the binary image
- **Region Labeling**: Label each distinct coin region for individual processing

### 7. Bounding Box Detection
- **Coordinate Extraction**: Determine minimum and maximum x,y coordinates for each detected coin
- **Box Drawing**: Generate bounding rectangles around identified coin regions

## How to Run

To run this project, follow these steps:

1. **Install Python 3.x**: Download from [python.org](https://www.python.org/)

2. **Install Required Libraries**:
   ```bash
   pip install matplotlib numpy
   ```

3. **Run the Pipeline**:
   ```bash
   python coin_detection.py input_image.jpg
   ```
