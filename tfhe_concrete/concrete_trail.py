import time
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision
import torchvision.transforms as transforms
import numpy as np

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from concrete.ml.torch.compile import compile_torch_model

#hyperparams
no_epochs = 2
batch_size = 32
learning_rate = 0.001

# class ConvolutionalNeuralNet(nn.Module):
#     def __init__(self, n_classes) -> None:
#         """Construct the CNN with a configurable number of classes."""
#         super().__init__()

#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
#         self.fc1 = nn.Linear(64 * 3 * 3, 64)
#         self.fc2 = nn.Linear(64, n_classes)

#     def forward(self, x):
#         X= self.conv1(x)
#         x = functional.relu(X)
#         x = self.pool(x)
#         X= self.conv2(x)
#         x = functional.relu(X)
#         x = self.pool(x)
#         X = self.conv3(x)
#         x = functional.relu(X)
#         x = torch.flatten(x, 1)
#         #x = x.view(-1, 64 * 3 * 3)
#         X = self.fc1(x)
#         x = functional.relu(X)
#         x = self.fc2(x)
#         return x
class ConvolutionalNeuralNet(nn.Module):
    def __init__(self, n_classes) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=3)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x= self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = x.flatten(1)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

#device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5),(0.5))     
        ])


#data sets downloading and reading
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                        train=True,
                                        download=True,
                                        transform=transform
                                        )

test_dataset = torchvision.datasets.MNIST(root='data',
                                        train=False,
                                        download=True,
                                        transform=transform
                                        )

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                        train=True,
                                        download=True,
                                        transform=transform
                                        )

train_loader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True
                            )

test_loader = torch.utils.data.DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False
                                    )

mnist_features = train_dataset.data.numpy().reshape(-1, 28, 28)
mnist_labels = train_dataset.targets.numpy()

# Reshape and expand dimensions to match the structure of load_digits dataset
x_train_mnist = np.expand_dims(mnist_features, 1)
x_train, x_test, y_train, y_test = train_test_split(
    x_train_mnist, mnist_labels,  train_size=1200, test_size=1000, shuffle=True, random_state=42
)
x_train = x_train.astype('float64')

net = ConvolutionalNeuralNet(10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

#Work now with concrete

def test_with_concrete(quantized_module, test_loader, use_sim):
    """Test a neural network that is quantized and compiled with Concrete ML."""

    # Casting the inputs into int64 is recommended
    all_y_pred = np.zeros((len(test_loader)), dtype=np.int64)
    all_targets = np.zeros((len(test_loader)), dtype=np.int64)

    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
    idx = 0
    for data, target in tqdm(test_loader):
        data = data.numpy()
        target = target.numpy()

        fhe_mode = "simulate" if use_sim else "execute"
        y_pred = quantized_module.forward(data, fhe=fhe_mode)
        endidx = idx + target.shape[0]
        all_targets[idx:endidx] = target

        # Get the predicted class id and accumulate the predictions
        y_pred = np.argmax(y_pred, axis=1)
        all_y_pred[idx:endidx] = y_pred

        # Update the index
        idx += target.shape[0]

    # Compute and report results
    n_correct = np.sum(all_targets == all_y_pred)

    return n_correct / len(test_loader)

n_bits = 6

q_module = compile_torch_model(
    net, 
    x_train, 
    rounding_threshold_bits=6, 
    p_error=0.1, 
    verbose=True,
    configuration=None
)
#q_module = compile_brevitas_qat_model(net, x_train)
start_time = time.time()
accs = test_with_concrete(
    q_module,
    train_loader,
    use_sim=True,
)
sim_time = time.time() - start_time

print(f"Simulated FHE execution for {n_bits} bit network accuracy: {accs:.2f}%")

# Generate keys first
t = time.time()
q_module.fhe_circuit.keygen()
print(f"Keygen time: {time.time()-t:.2f}s")


t = time.time()
accuracy_test = test_with_concrete(
    q_module,
    test_loader,
    use_sim=False,
)
elapsed_time = time.time() - t
time_per_inference = elapsed_time / len(test_loader)
accuracy_percentage = 100 * accuracy_test

print(
    f"Time per inference in FHE: {time_per_inference:.2f} "
    f"with {accuracy_percentage:.2f}% accuracy"
)