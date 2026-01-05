from torchvision import datasets, transforms
from pathlib import Path
import ssl


# download MNIST if not already downloaded, then return original train-test splits
def get_mnist_datasets(data_dir: str = "./data"):
    data_dir = Path(data_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),  # convert to [0,1]
    ])

    ssl._create_default_https_context = ssl._create_unverified_context
    mnist_train = datasets.MNIST(
        root=data_dir,
        train=True,
        download=not (data_dir / "MNIST").exists(),
        transform=transform
    )

    mnist_test = datasets.MNIST(
        root=data_dir,
        train=False,
        download=not (data_dir / "MNIST").exists(),
        transform=transform
    )

    return mnist_train, mnist_test
