# OCR Preprocessing Pipeline Tasks




## Task 4: Deskewing and Rotation Correction

- Detect the dominant text orientation angle
- Rotate the image to correct text skew and make it horizontal

## Task 5: Morphological Operations for Noise Removal and Text Repair

- Apply morphological opening to remove small isolated noise pixels
- Optionally use closing to repair faint/broken text shapes without merging characters







## Task 6: Contrast Enhancement (Optional)

- Enhance contrast using normalization or histogram equalization if needed to improve text visibility

## Task 7: Standardize Image Size and Resolution

- Resize or rescale images to a consistent size and resolution (e.g., 300 DPI) for uniform input quality

## Task 8: Cropping and Border Removal (If Needed)

- Crop out irrelevant borders or padding to focus on text region and avoid distractions

## Task 9: Quality Inspection and Saving

- Visually verify processed images for noise, skew, blurriness, or text degradation
- Save final preprocessed images in a lossless format like PNG to prevent compression artifacts
