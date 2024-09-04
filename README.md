# Spontaneously functionalized convolutional neural network project

This project aims to investigate whether convolutional neural networks (CNNs) can spontaneously form specialized functional areas in a multi-task learning environment, similar to the fusiform face area (FFA) of the human brain. Through training and lesion experiments, we explored the performance of VGG16, VGG19, ResNet50, and ResNeXt50 models when performing both face and non-face image recognition tasks.

## Project Structure

- `vgg16.py`：A multi-task neural network was built using the VGG16 model, including face and non-face image recognition. A lesion experiment was then conducted to verify whether the network spontaneously functionalized. If the human brain also has a functional area dedicated to recognizing face images (fusiform face area).
- `vgg19.py`：The VGG19 model is used, which has similar structure and function to VGG16, but increases the network depth to improve the recognition ability of the multi-task network. Similar lesion experiments are still conducted.
- `ResNet.py`：The ResNet50 model is used to improve the recognition effect of the multi-task network through residual learning. And through lesion experiments, it is further verified that the ResNet model will also spontaneously show certain functionalization.
- `ResNeXt.py`：The ResNeXt50 model is used to enhance the expressiveness and scalability of the multi-task network through group convolution. The lesion experiment is still used to verify whether the model spontaneously presents functional partitioning.
- `vggface2.py`：Data processing scripts to prepare and preprocess the VGGFace2 dataset and the CIFAR-10 dataset for training in a unified format.
- **Lesion Experiment Section**：Each model file contains code for impairment experiments designed to assess the functionality of specific parts of the network (e.g., a specific convolutional layer).

## Dataset

- **VGGFace2 Dataset**：A large-scale face recognition dataset used to train models to recognize faces.
- **CIFAR-10 Dataset**：A standard image recognition dataset consisting of 10 categories of natural images, such as airplanes, cars, cats, etc.

## Environment Setup

Please make sure to install the following main dependencies before running the code:

- Python 3.x
- NumPy
- Torch
- Torchvision
- scikit-learn
- Pillow
- Matplotlib
- Seaborn
- OpenCV

These dependencies can be installed using the following command:

```bash
pip install numpy torch torchvision scikit-learn pillow matplotlib seaborn opencv-python
