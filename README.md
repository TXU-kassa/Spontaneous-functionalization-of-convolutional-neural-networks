# 自发功能化的卷积神经网络项目

本项目旨在研究卷积神经网络（CNN）在多任务学习环境中能否自发形成专门的功能区域，类似于人脑的面孔区域（Fusiform Face Area, FFA）。通过训练和损伤实验，我们探讨了 VGG16、VGG19、ResNet50 和 ResNeXt50 模型在同时进行人脸和非人脸图像识别任务时的表现。

## 项目结构

- `vgg16.py`：使用 VGG16 模型进行图像分类任务，包括人脸和非人脸图像识别。
- `vgg19.py`：使用 VGG19 模型，结构和功能类似于 VGG16，但增加了网络深度以提高识别能力。
- `ResNet.py`：使用 ResNet50 模型，通过残差学习提高深度网络的训练效果。
- `ResNeXt.py`：使用 ResNeXt50 模型，通过分组卷积的方式增强模型的表达能力和扩展性。
- `vggface2.py`：数据处理脚本，用于准备和预处理 VGGFace2 数据集以及 CIFAR-10 数据集，以统一格式进行训练。
- **损伤实验部分**：每个模型文件中都包含损伤实验代码，旨在评估网络特定部分（如特定卷积层）的功能化程度。

## 数据集

- **VGGFace2 数据集**：一个大规模人脸识别数据集，用于训练模型识别人脸。
- **CIFAR-10 数据集**：一个标准的图像识别数据集，包括10个类别的自然图像，如飞机、汽车、猫等。

## 环境设置

请确保在运行代码之前，安装以下主要依赖项：

- Python 3.x
- NumPy
- Torch
- Torchvision
- scikit-learn
- Pillow
- Matplotlib
- Seaborn
- OpenCV

可以使用以下命令安装这些依赖项：

```bash
pip install numpy torch torchvision scikit-learn pillow matplotlib seaborn opencv-python
