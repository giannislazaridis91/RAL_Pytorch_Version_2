import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class Dataset:

    def __init__(self, number_of_state_data, train_dataset_length):
        self.train_data = np.array([[]])
        self.train_labels = np.array([[]])
        self.train_labels_one_hot_encoding = np.array([[]])
        self.test_data = np.array([[]])
        self.test_labels = np.array([[]])
        self.test_labels_one_hot_encoding = np.array([[]])
        self.state_data = np.array([[]])
        self.state_labels = np.array([[]])
        self.state_labels_one_hot_encoding = np.array([[]])
        self.number_of_state_data = number_of_state_data
        self.number_of_test_data = 10000
        self.number_of_classes = 0
        self.train_dataset_length = train_dataset_length
        self.regenerate()

    def regenerate(self):
        # Placeholder function for regenerating the dataset.
        pass

    def _normalization(self):
        # Data normalization to [0, 1]
        self.train_data = self.train_data.astype('float32') / 255.0
        self.test_data = self.test_data.astype('float32') / 255.0
        self.state_data = self.state_data.astype('float32') / 255.0

    def _one_hot_encoding(self):
        # Convert class vectors to binary class matrices using one-hot encoding.
        self.train_labels_one_hot_encoding = F.one_hot(torch.tensor(self.train_labels),
                                                       num_classes=self.number_of_classes).float().numpy()
        self.test_labels_one_hot_encoding = F.one_hot(torch.tensor(self.test_labels),
                                                      num_classes=self.number_of_classes).float().numpy()
        self.state_labels_one_hot_encoding = F.one_hot(torch.tensor(self.state_labels),
                                                       num_classes=self.number_of_classes).float().numpy()


class DatasetCIFAR10(Dataset):

    def __init__(self, number_of_state_data, train_dataset_length):
        super().__init__(number_of_state_data, train_dataset_length)

    def regenerate(self):
        # Load CIFAR-10 dataset using torchvision
        transform = transforms.Compose([transforms.ToTensor()])
        cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        train_data = np.array([np.array(img) for img, _ in cifar10_train])
        train_labels = np.array([label for _, label in cifar10_train])
        test_data = np.array([np.array(img) for img, _ in cifar10_test])
        test_labels = np.array([label for _, label in cifar10_test])

        # Train data.
        new_data = []
        new_data_labels = []
        classes = list(range(10))
        for data_class in classes:
            count = 0
            for i in range(len(train_labels)):
                if int(train_labels[i]) == data_class and count < self.train_dataset_length / 10:
                    count += 1
                    new_data.append(train_data[i])
                    new_data_labels.append(train_labels[i])
        self.train_data = np.array(new_data)
        self.train_labels = np.array(new_data_labels)

        # Test data.
        new_data = []
        new_data_labels = []
        for data_class in classes:
            count = 0
            for i in range(len(test_labels)):
                if int(test_labels[i]) == data_class and count < (self.number_of_test_data / 10):
                    count += 1
                    new_data.append(test_data[i])
                    new_data_labels.append(test_labels[i])
        self.test_data = np.array(new_data)
        self.test_labels = np.array(new_data_labels)

        # State data.
        new_data = []
        new_data_labels = []
        for data_class in classes:
            count = 0
            for i in range(len(test_labels)):
                if int(test_labels[-i]) == data_class and count < (self.number_of_state_data / 10):
                    count += 1
                    new_data.append(test_data[-i])
                    new_data_labels.append(test_labels[-i])
        self.state_data = np.array(new_data)
        self.state_labels = np.array(new_data_labels)

        self.number_of_classes = len(np.unique(self.train_labels))
        self._normalization()
        self._one_hot_encoding()