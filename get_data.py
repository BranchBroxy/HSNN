import torch
import torchvision
import torchvision.transforms as transforms

def load_mnist(batch_size):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    data_path = './data'
    # Define a transform
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)
    return train_loader, test_loader

def add_noise_to_mnist_dataset(dataset, noise_level):
    noisy_dataset = []
    for data in dataset:
        image, label = data
        # Add noise to the image
        image = image + noise_level * torch.randn(image.size())
        # Clip the image to be between 0 and 1
        image = torch.clamp(image, 0, 1)
        # Add the noisy data to the new dataset
        noisy_dataset.append((image, label))
    return noisy_dataset

