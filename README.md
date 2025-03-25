# Image Processing Application

## Overview
This project is a raster image processing application developed as part of the "Biometrics" course at Warsaw University of Technology. The application provides a variety of image processing operations, including grayscale conversions, filters, edge detection, brightness/contrast adjustments, and histogram visualization.

## Features
- **Image Loading/Saving**: Supports common formats (PNG, JPG, JPEG, BMP)
- **Pixel Operations**:
  - Grayscale conversions (Average, Lightness, Desaturation)
  - Negative image
  - Binarization with adjustable threshold
- **Filters**:
  - Sharpening
  - Blur (with parameter control)
  - Gaussian filter (adjustable sigma)
  - Laplacian edge detection
  - Gradient operators (Sobel, Prewitt, Roberts, Scharr, Sobel-Feldman)
- **Adjustments**:
  - Brightness control (-255 to 255)
  - Contrast control (-255 to 255)
- **Custom Filters**: 3x3 kernel with user-defined values
- **Visualization**:
  - Side-by-side original/processed image display
  - RGB and grayscale histograms
  - Horizontal/vertical projections
- **History Management**: Undo functionality and reset option

## Application View
![Application View](./examples%20of%20usage/app.png)


## Usage
1. Load an image using the "Load image" button
2. Select an operation from the dropdown menu
3. Adjust parameters if needed (threshold, brightness, contrast, etc.)
4. Click "Apply" to process the image
5. Use "Undo" to revert changes or "Reset" to return to the original
6. Save processed images with the "Save" button

## Authors
- [Maciej Momot](https://github.com/MaciejMomot)  
- [Filip Szlingiert](https://github.com/FylypO)