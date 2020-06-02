# Glomeruli Mask Annotation using Image Processing

- Automated annotation of Glomeruli in cells using image pre processing
- Increasing contrast, blur and thresholding operation is applied to create mask of given tile
- Binary mask was passed to output json object which can be used for training Glomeruli detection model using Deep Learning methods
- For a given cell image, tiles of given length was cropped and then whole operation was applied to create unique mask