import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg19
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import cv2
import time

start_time = time.time()


# Unified vgg19 Model
class UnifiedVGG(nn.Module):
    def __init__(self, num_classes):
        super(UnifiedVGG, self).__init__()
        self.base = vgg19(pretrained=True)
        # Do not use the last fully connected layer of the pre-trained model
        self.base.classifier = nn.Identity()  # Remove the last fully connected layer of VGG
        # Create a new classifier to suit your task, with input features of 512 * 7 * 7 (the flattened result after avgpool from VGG19)
        self.classifier = nn.Linear(512 * 7 * 7, 20)
        # self.classifier = nn.Linear(512 * 7 * 7, 12)

    def forward(self, x):
        x = self.base(x)  # This will output the results after avgpool
        x = torch.flatten(x, 1)  # Flatten
        output = self.classifier(x)  # Pass the new classifier
        return output


def check_labels(dataset):
    labels = [label for _, label in dataset]
    print("Label range:", min(labels), "to", max(labels))
    assert min(labels) >= 0 and max(labels) < len(class_names), "Labels out of range."


class CombinedDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                self.data.append(batch[b'data'])

                # Check tag type and convert appropriately
                batch_labels = batch[b'labels']
                # Increase 10 to distinguish the labels of the CIFAR-10 dataset and the face dataset
                transformed_labels = [label + 10 if "vggface2_cifar" in file_path else label for label in batch_labels]

                self.labels.extend(transformed_labels)

        self.data = np.concatenate(self.data)
        self.data = self.data.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            # Make sure img is of the appropriate data type
            img = img.astype(np.float32)  # Confirm that img is a floating point type
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to 0-1
            img = (img * 255).astype(np.uint8)  # Convert to uint8 type with range 0-255
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)



# Define data conversion
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Add random horizontal flip
    transforms.RandomRotation(10),  # Add random rotation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

file_paths = ['cifar-10-batches-py/data_batch_1', 'cifar-10-batches-py/data_batch_2',
              'cifar-10-batches-py/data_batch_3', 'cifar-10-batches-py/data_batch_4',
              'cifar-10-batches-py/data_batch_5',
              'vggface2_cifar/data_batch_1',
              'vggface2_cifar/data_batch_2', 'vggface2_cifar/data_batch_3',
              'vggface2_cifar/data_batch_4', 'vggface2_cifar/data_batch_5']

dataset = CombinedDataset(file_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

file_paths_test = ['cifar-10-batches-py/test_batch', 'vggface2_cifar/test_batch']
dataset_text = CombinedDataset(file_paths_test, transform=transform)
dataloader_text = DataLoader(dataset_text, batch_size=16, shuffle=True)


# Training Function
def train(model, dataloader, criterion, optimizer, device, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}')


# Evaluation Function

def evaluate(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate and print the accuracy of each category
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for class_name, accuracy in zip(class_names, class_accuracies):
        print(f"{class_name} Accuracy: {accuracy:.2f}")

    # Calculate the overall accuracy of object and face categories separately
    object_accuracy = accuracy_score(
        [label for label in all_labels if label < 10],
        [pred for label, pred in zip(all_labels, all_preds) if label < 10]
    )
    face_accuracy = accuracy_score(
        [label for label in all_labels if label >= 10],
        [pred for label, pred in zip(all_labels, all_preds) if label >= 10]
    )
    print("Overall Object Accuracy: {:.2f}".format(object_accuracy))
    print("Overall Face Accuracy: {:.2f}".format(face_accuracy))

    # Print classification report, including precision, recall and F1 score
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    return class_accuracies, object_accuracy, face_accuracy

def evaluate1(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)

    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    object_accuracy = accuracy_score(
        [label for label in all_labels if label < 10],
        [pred for label, pred in zip(all_labels, all_preds) if label < 10]
    )
    face_accuracy = accuracy_score(
        [label for label in all_labels if label >= 10],
        [pred for label, pred in zip(all_labels, all_preds) if label >= 10]
    )
    print("Overall Object Accuracy: {:.2f}".format(object_accuracy))
    print("Overall Face Accuracy: {:.2f}".format(face_accuracy))
    return class_accuracies, object_accuracy, face_accuracy


# Define class names according to your labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck',
               'id_1','id_2','id_3','id_4','id_5','id_6','id_7','id_8','id_9','id_10']

def lesion_study_by_group(model, layer_index, dataloader, device, class_names, group_size=256):
    num_filters = model.base.features[layer_index].weight.size(0)
    num_groups = num_filters // group_size
    filter_impacts = []

    for g in range(num_groups):
        original_weights = model.base.features[layer_index].weight.data.clone()
        start_idx = g * group_size
        end_idx = start_idx + group_size

        # Damage the filter of the current group: reset the weights to zero
        with torch.no_grad():
            model.base.features[layer_index].weight.data[start_idx:end_idx] = 0.0

        # Use the evaluation function to evaluate the performance after damage
        _, object_accuracy, face_accuracy = evaluate1(model, dataloader, device, class_names)


        filter_impacts.append((g, object_accuracy, face_accuracy))


        model.base.features[layer_index].weight.data = original_weights

    # Sort by performance after damage, you can choose the key indicators for sorting
    filter_impacts.sort(key=lambda x: x[1], reverse=True)

    # Print the impact of each set of filters
    for impact in filter_impacts:
        print(f"Group {impact[0]}: Object Accuracy = {impact[1]:.2f}, Face Accuracy = {impact[2]:.2f}")

    return filter_impacts


def extract_balanced_features(model, dataloader, device, layer_index, max_samples_per_class):
    model.eval()
    features_objects = []
    features_faces = []
    labels_objects = []
    labels_faces = []

    count_objects = [0] * 10
    count_faces = [0] * 10

    def hook(module, input, output):
        output_flat = output.detach().cpu().numpy().reshape(output.size(0), -1)
        for feature, label in zip(output_flat, labels):
            if label < 10 and count_objects[label] < max_samples_per_class:
                features_objects.append(feature)
                labels_objects.append(label)
                count_objects[label] += 1
            elif label >= 10 and count_faces[label-10] < max_samples_per_class:
                features_faces.append(feature)
                labels_faces.append(label)
                count_faces[label-10] += 1

    handle = model.base.features[layer_index].register_forward_hook(hook)

    for images, labels in dataloader:
        if all(c >= max_samples_per_class for c in count_objects) and all(c >= max_samples_per_class for c in count_faces):
            break
        images = images.to(device)
        labels = labels.cpu().numpy()
        _ = model(images)

    handle.remove()

    features_objects = np.array(features_objects)
    features_faces = np.array(features_faces)
    labels_objects = np.array(labels_objects)
    labels_faces = np.array(labels_faces)

    return (features_objects, labels_objects), (features_faces, labels_faces)


def plot_pca(features_objects, features_faces):
    pca = PCA(n_components=2)
    objects_pca = pca.fit_transform(features_objects)
    faces_pca = pca.fit_transform(features_faces)

    plt.figure(figsize=(10, 5))
    plt.scatter(objects_pca[:, 0], objects_pca[:, 1], c='blue', label='Objects', alpha=0.5)
    plt.scatter(faces_pca[:, 0], faces_pca[:, 1], c='red', label='Faces', alpha=0.5)
    plt.legend()
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization of Object and Face Features')
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnifiedVGG(num_classes=len(class_names))
model.to(device)
print(model)
def check_labels(dataloader):
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels.numpy())

    unique_labels = np.unique(all_labels)
    return unique_labels

# Check using data loader
unique_labels = check_labels(dataloader)
print("Unique labels in the dataset:", unique_labels)
print("Number of unique labels:", len(unique_labels))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train and Evaluate
train(model, dataloader, criterion, optimizer, device)
evaluate(model, dataloader_text, device, class_names)
layer_to_lesion = 34
lesion_study_by_group(model, layer_to_lesion, dataloader_text, device, class_names)
(features_objects, labels_objects), (features_faces, labels_faces) = extract_balanced_features(
    model, dataloader_text, device, layer_to_lesion, max_samples_per_class=100)
plot_pca(features_objects, features_faces)
end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")
