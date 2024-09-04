import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

start_time = time.time()


# Unified ResNeXt Model
class UnifiedResNeXt(nn.Module):
    def __init__(self, num_classes):
        super(UnifiedResNeXt, self).__init__()
        self.base = resnext50_32x4d(pretrained=True)
        # Replace the final fully connected layer to suit your classification task
        num_features = self.base.fc.in_features
        self.base.fc = nn.Linear(num_features, 20)  # Adjust to the desired number of categories

    def forward(self, x):
        return self.base(x)



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

                batch_labels = batch[b'labels']

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
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)



# Define data conversion
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
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

    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for class_name, accuracy in zip(class_names, class_accuracies):
        print(f"{class_name} Accuracy: {accuracy:.2f}")

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

def lesion_by_group(model, dataloader, device, class_names):
    last_bottleneck = model.base.layer4[-1]
    last_conv_layer = last_bottleneck.conv3
    num_filters = last_conv_layer.weight.size(0)
    group_size = num_filters // 32

    original_weights = last_conv_layer.weight.data.clone()

    accuracies = []

    # Cycle damage to each filter group
    for i in range(32):
        # Damage the filter of the current group: reset the weights to zero
        with torch.no_grad():
            start = i * group_size
            end = start + group_size
            last_conv_layer.weight.data[start:end] = 0.0

        # Evaluate model performance after damage
        _, object_accuracy, face_accuracy = evaluate1(model, dataloader, device, class_names)

        # Record and print the results
        accuracies.append((object_accuracy, face_accuracy))
        print(f"Group {i+1} - Object Accuracy after lesion: {object_accuracy:.2f}, Face Accuracy: {face_accuracy:.2f}")

        # Restore weights
        last_conv_layer.weight.data = original_weights

    return accuracies

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnifiedResNeXt(num_classes=len(class_names))
model.to(device)
print(model)
def check_labels(dataloader):
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels.numpy())

    unique_labels = np.unique(all_labels)
    return unique_labels


unique_labels = check_labels(dataloader)
print("Unique labels in the dataset:", unique_labels)
print("Number of unique labels:", len(unique_labels))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train and Evaluate
train(model, dataloader, criterion, optimizer, device)
evaluate(model, dataloader_text, device, class_names)
lesion_by_group(model, dataloader_text, device, class_names)

end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")
