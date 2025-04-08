# Multi-Level Wavelet Decomposition and Image Compression (MATLAB)

This project implements a multi-level wavelet decomposition and compression algorithm for images using MATLAB. It supports both grayscale and color images, providing flexibility in compression levels and visual quality.

## Project Overview

This repository contains:

* **MATLAB Scripts:** Implementation of the image compression algorithm, including wavelet decomposition, quantization, and Huffman encoding/decoding.
* **Example Images:** Sample grayscale and color images used for testing.
* **Report:** A detailed report explaining the methodology, results, and conclusions of the project.

## Development Process (SDLC)

This project was developed with a structured approach based on the Software Development Life Cycle:

**1. Planning and Requirements Gathering:**
* Objective: Design and implement a subband (wavelet) image coding system in MATLAB for advanced image compression.
* Stakeholders: Myself (as the student/developer).
* Requirements: Input of BMP and TIF images, implementation of wavelet decomposition, quantization, and Huffman coding, achievement of specific compression and quality goals, and user-defined quantization.
* Project Plan: Followed a structured plan based on course requirements and design considerations.

**2. Design:**
* System Architecture: Encoder (Decomposition -> Quantization -> Entropy Encoding) and Decoder (Entropy Decoding -> Inverse Quantization -> Reconstruction).
* Algorithm Design: Selected specific wavelet transforms, quantization methods, and Huffman coding.
* User Interface (Basic): Implemented user control over quantization step sizes via script variables.

**3. Development (or Implementation):**
* Coding MATLAB scripts for the encoder and decoder functions.
* Utilizing MATLAB's image processing and signal processing toolboxes.

**4. Testing:**
* Unit testing of individual functions (e.g., wavelet decomposition, quantization).
* Integration testing of the complete encoder-decoder system.
* System testing to evaluate compression ratios and PSNR for various images and settings.
* Implicit "user acceptance testing" through visual inspection of reconstructed images.

**5. Deployment:**
* Deployment involved submitting the MATLAB code and a project report.
* Provided basic instructions on how to run the scripts and adjust parameters.

**6. Maintenance (For a Course Project):**
* Addressed bugs encountered during development and testing.
* Considered potential enhancements for future exploration.
* Documented the code with comments and provided a project report.

## Functionality

The MATLAB scripts perform the following operations:

1.  **Image Loading:** Loads grayscale (.bmp) or color (.tiff) images.
2.  **Multi-Level Wavelet Decomposition:** Decomposes the image into subbands using a 2D wavelet transform.
3.  **Scalar Quantization:** Quantizes the wavelet coefficients using variable step sizes.
4.  **Entropy Encoding (Huffman):** Encodes the quantized coefficients using Huffman coding.
5.  **Entropy Decoding (Huffman):** Decodes the Huffman-encoded data.
6.  **Dequantization:** Dequantizes the decoded coefficients.
7.  **Multi-Level Wavelet Reconstruction:** Reconstructs the image from the dequantized coefficients.
8.  **Performance Evaluation:** Calculates PSNR (Peak Signal-to-Noise Ratio) and compression ratios.
9.  **Data Storage:** Saves compressed data and metadata for reconstruction.

## Requirements

* MATLAB (with Image Processing Toolbox)

## Usage

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/your-username/Subband-Image-Coding-Implementation.git](https://github.com/your-username/Subband-Image-Coding-Implementation.git)
    cd Subband-Image-Coding-Implementation
    ```

2.  **Open MATLAB:**
    * Launch MATLAB.

3.  **Navigate to the Project Directory:**
    * In the MATLAB command window, navigate to the directory where you cloned the repository.

4.  **Run the `cpeProject.m` Script:**
    * Execute the `cpeProject.m` script.
    * You can modify the `input_image_path`, `color`, `decomposition_levels`, and `quantization_steps` variables within the script to adjust the compression process.

5.  **View Results:**
    * The script will output the reconstructed images and performance metrics (PSNR, compression ratios).
    * Reconstructed images are saved as `.png` files.

## File Structure

Subband-Image-Coding-Implementation/
├── cpeProject.m             - Main script for image compression.
├── lena_color.tiff          - Example color image.
├── lena_gray.bmp           - Example grayscale image.
├── Report.docx               - Project report.
├── bin files/                - Directory for compressed binary data.
├── meta files/               - Directory for metadata.
├── png files/                - Directory for reconstructed PNG images.
└── README.md


## Customization

* **Input Image:** Change the `input_image_path` to process different images.
* **Image Type:** Set `color` to `0` for grayscale and `1` for color images.
* **Decomposition Levels:** Adjust `decomposition_levels` to control the depth of wavelet decomposition.
* **Quantization Steps:** Modify `quantization_steps` to experiment with different compression levels.
* **Output Format:** Change the `imwrite` format in the script to save reconstructed images in other formats.

## Key Functions

* `multi_level_wavelet_decomposition`: Performs multi-level wavelet decomposition.
* `perform_2d_wavelet_decomposition`: Performs 2D wavelet decomposition.
* `multi_level_wavelet_reconstruction`: Reconstructs the image from wavelet coefficients.
* `perform_2d_inverse_wavelet_reconstruction`: Performs 2D inverse wavelet reconstruction.
* `huffman_encode`: Encodes data using Huffman coding.
* `huffman_decode`: Decodes Huffman-encoded data.
* `calculate_psnr`: Calculates the Peak Signal-to-Noise Ratio.
* `calculate_psnr_values`: Calculates PSNR for different quantization steps.
* `calculate_compression_ratios`: Calculates compression ratios for different quantization steps.
* `save_color_compressed_data`: Saves compressed data and metadata.
* `load_color_compressed_data`: Loads compressed data and metadata.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.
