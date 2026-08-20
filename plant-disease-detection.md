# Plant Disease Detection (CNN Image Classifier)

**Notebook:** [`PlantDiseaseDetection.ipynb`](./PlantDiseaseDetection.ipynb)
**Tools:** Python · TensorFlow/Keras · OpenCV · scikit-learn

## Objective

Build a convolutional neural network that classifies plant leaf images into one of 39
disease/healthy categories from the PlantVillage dataset, as a foundation for automated crop
disease detection.

## Approach

1. **Data pipeline** — downloaded and unzipped the ~866MB PlantVillage image dataset, loaded
   and resized images to 256×256 with OpenCV, converted them to normalized NumPy arrays, and
   encoded the 39 class labels with `LabelBinarizer`.
2. **Data augmentation** — applied random rotation, width/height shift, shear, zoom, and
   horizontal flip via Keras `ImageDataGenerator` to improve generalization given a limited
   image set.
3. **Model architecture** — a custom CNN built with Keras `Sequential`: three convolutional
   blocks (32 → 64 → 128 filters), each with batch normalization, max pooling, and dropout,
   followed by a 1,024-unit dense layer and a 39-way softmax output (~58M trainable
   parameters).
4. **Training** — trained for 25 epochs with the Adam optimizer and binary cross-entropy loss,
   batch size 32, on an 80/20 train/test split.
5. **Result** — reached **~54% accuracy** on the held-out test set. As a from-scratch CNN
   (no transfer learning), this establishes a baseline; the clear next step is applying
   transfer learning (e.g. MobileNet or ResNet pretrained on ImageNet) to improve accuracy.

## Skills Demonstrated

CNN architecture design, image preprocessing pipelines, data augmentation, multi-class image
classification, and GPU-accelerated model training.
