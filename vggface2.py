import os
import numpy as np
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_images_and_labels(base_dir, classes):
    images, labels = [], []
    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(base_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            images.append(transform(image))
            labels.append(idx)
    return images, labels


def create_batches(data, labels, num_train_batches):
    # Convert the data into a form suitable for model input
    data = torch.stack(data).view(len(data), -1)  # Convert to a one-dimensional array
    labels = torch.tensor(labels)

    # Shuffle the data
    indices = torch.randperm(len(data))
    data, labels = data[indices], labels[indices]

    # Total number of samples
    num_samples = len(data)
    # Calculate the size of the training set and test set
    train_size = int(num_samples * 0.8)  # 80% of the data is used for training
    test_size = num_samples - train_size  # 20% of the data is used for testing

    # To ensure there are only five training batches
    batch_size = train_size // num_train_batches
    remainder = train_size % num_train_batches

    train_batches = []
    start_idx = 0
    for i in range(num_train_batches):
        end_idx = start_idx + batch_size + (1 if i < remainder else 0)
        train_batches.append((data[start_idx:end_idx], labels[start_idx:end_idx]))
        start_idx = end_idx

    test_batch = (data[train_size:], labels[train_size:])
    return train_batches, test_batch


def save_batch(data, labels, output_file):
    images = data.numpy()
    labels = labels.numpy()
    data_dict = {
        b'data': images,
        b'labels': labels
    }
    print(f"Saving batch to {output_file} with {len(images)} images.")
    with open(output_file, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    base_dir = 'vggface2'
    classes = sorted(os.listdir(base_dir))
    images, labels = load_images_and_labels(base_dir, classes)

    train_batches, test_batch = create_batches(images, labels, 5)  # Create batches using your function

    save_dir = 'vggface2_cifar'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save training batches
    for i, (batch_data, batch_labels) in enumerate(train_batches):
        save_batch(batch_data, batch_labels, os.path.join(save_dir, f'data_batch_{i + 1}'))

    # Save the test batch
    save_batch(*test_batch, os.path.join(save_dir, 'test_batch'))


if __name__ == "__main__":
    main()
