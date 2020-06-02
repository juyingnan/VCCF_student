# Nuclei Segmentation using Mask RCNN
This is an implementation of Mask R-CNN on Python 3, Keras, and TensorFlow. The model detects nuclei using instance segmentation method. It generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet50 backbone.

The repository includes:

- Source code of Mask R-CNN.
- Jupyter notebooks to illustrate the training and detection pipeline at every step.
- Additional test data.

Changes:

- nucleus.py: Loading custom dataset for training, testing or prediction

Dataset: https://www.kaggle.com/c/data-science-bowl-2018/data

Weights: https://bit.ly/MaskRCNN_Weights

### References:
```latex
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```
