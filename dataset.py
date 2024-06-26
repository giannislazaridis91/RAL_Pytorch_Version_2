import torch
from torch.utils.data import Dataset
from torchvision import datasets

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, number_of_state_data, number_of_warm_start_data, number_of_agent_data, number_of_test_methods_data, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.number_of_state_data = number_of_state_data
        self.number_of_warm_start_data = number_of_warm_start_data
        self.number_of_agent_data = number_of_agent_data
        self.number_of_test_methods_data = number_of_test_methods_data
        self.regenerate()
        
    def regenerate(self):


        # Load CIFAR10 dataset
        train_dataset = datasets.CIFAR10(root=self.root_dir, train=True, download=True, transform=None)
        train_data = train_dataset.data
        train_labels = train_dataset.targets

        # Split train_data and train_labels to subsets per class
        class_data = [[] for _ in range(10)]
        class_labels = [[] for _ in range(10)]
        
        for i in range(len(train_labels)):
            class_data[train_labels[i]].append(train_data[i])
            class_labels[train_labels[i]].append(train_labels[i])

        # Sample subset data from each class based on provided numbers
        state_data = []
        state_labels = []
        warm_start_data = []
        warm_start_labels = []
        agent_data = []
        agent_labels = []
        test_methods_data = []
        test_methods_labels = []

        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        for data_class in classes:
            for i in range(int(self.number_of_state_data / 10)):
                state_data.append(class_data[data_class][i])
                state_labels.append(class_labels[data_class][i])
                
            for i in range(int(self.number_of_state_data / 10), int(self.number_of_state_data / 10) + int(self.number_of_warm_start_data / 10)):
                warm_start_data.append(class_data[data_class][i])
                warm_start_labels.append(class_labels[data_class][i])
                
            for i in range(int(self.number_of_state_data / 10) + int(self.number_of_warm_start_data / 10), int(self.number_of_state_data / 10) + int(self.number_of_warm_start_data / 10) + int(self.number_of_agent_data / 10)):
                agent_data.append(class_data[data_class][i])
                agent_labels.append(class_labels[data_class][i])
                
            for i in range(int(self.number_of_state_data / 10) + int(self.number_of_warm_start_data / 10) + int(self.number_of_agent_data / 10), int(self.number_of_state_data / 10) + int(self.number_of_warm_start_data / 10) + int(self.number_of_agent_data / 10) + int(self.number_of_test_methods_data / 10)):
                test_methods_data.append(class_data[data_class][i])
                test_methods_labels.append(class_labels[data_class][i])

        # Convert lists to numpy arrays and then to PyTorch tensors
        self.state_data = torch.tensor(state_data)
        self.warm_start_data = torch.tensor(warm_start_data)
        self.agent_data = torch.tensor(agent_data)
        self.test_methods_data = torch.tensor(test_methods_data)

        self.state_labels = torch.tensor(state_labels)
        self.warm_start_labels = torch.tensor(warm_start_labels)
        self.agent_labels = torch.tensor(agent_labels)
        self.test_methods_labels = torch.tensor(test_methods_labels)

        self.test_data = torch.tensor(train_dataset.data)
        self.test_labels = torch.tensor(train_dataset.targets)

        self.number_of_classes = 10
        
        # Normalize and apply one-hot encoding (if needed)
        self._normalization()
        # self._one_hot_encoding()

    def _normalization(self):
        # Data normalization (assuming RGB images, divide by 255)
        self.state_data = self.state_data.float() / 255.0
        self.warm_start_data = self.warm_start_data.float() / 255.0
        self.agent_data = self.agent_data.float() / 255.0
        self.test_methods_data = self.test_methods_data.float() / 255.0
        self.test_data = self.test_data.float() / 255.0

    def __len__(self):
        return len(self.state_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'state_data': self.state_data[idx],
            'warm_start_data': self.warm_start_data[idx],
            'agent_data': self.agent_data[idx],
            'test_methods_data': self.test_methods_data[idx],
            'state_labels': self.state_labels[idx],
            'warm_start_labels': self.warm_start_labels[idx],
            'agent_labels': self.agent_labels[idx],
            'test_methods_labels': self.test_methods_labels[idx],
        }

        return sample