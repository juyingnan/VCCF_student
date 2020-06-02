# Glomeruli Detection using MaskRCNN (Bounding Box)

- Manually annotate train images in XML format that provide bounding boxes for glomeruli in cell images
- Create masks for glomeruli in cells and label them as 1 while the rest of the background as 0
- Create unique identifier of multiple masks in a single cell image
- Run MaskRCNN on the train images and load weights of pre trained mask_rcnn_coco model
- Evaluate the model on test images using Mean Average Precision value

## Parameters

- Train images : 100
- Test images : 19

## Evaluation metric

- Train mAP : 0.9
- Test mAP : 0.89
