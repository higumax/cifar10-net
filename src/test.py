import argparse
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
from torchvision import transforms
from torch.utils.data import DataLoader
from model import Classifier

device = "cuda" if torch.cuda.is_available() else "cpu"
cifar10_path = "data/cifar-10-batches-py"
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

def parseargs():
    parser = argparse.ArgumentParser(description="Classifier tester")
    parser.add_argument("--batch_size", "-b", type=int, default=100)
    parser.add_argument("--weight", "-w", type=str, default="")
    return parser.parse_args()

def test(args):
    batch_size = args.batch_size
    classifier_weight = args.weight

    # load test dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    X_train, y_train, X_test, y_test = utils.load_cifar10(cifar10_path)

    assert X_test.shape[0] == y_test.shape[0]
    print(f"Dataset (test) size = {y_test.shape[0]}")

    test_data = utils.MyDataset(x=X_test, y=y_test, transform=transform)
    #test_data = utils.MyDataset(x=X_train, y=y_train, transform=transform)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=1)


    # model, optimizer, loss function
    model = Classifier().to(device)
    model.load_state_dict(torch.load(classifier_weight))

    # test
    correct = 0
    total = 0
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward
            outputs, _ = model(inputs)

            _, predicted_labels = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

            y_pred += predicted_labels.cpu().numpy().tolist()
    
    print(f"Accuracy: {100 * correct / total:.2f}%")

    for i, label in enumerate(CIFAR10_CLASSES):
        correct = ((y_test == i)*1) * ((np.array(y_pred) == y_test)*1)
        class_accuracy = 100 * correct.sum() / y_test[y_test == i].shape[0]
        print(f"{label}: {class_accuracy:.1f}%")

    confusion_matrix = torch.zeros(len(CIFAR10_CLASSES), len(CIFAR10_CLASSES))
    for t, p in zip(y_test, y_pred):
        confusion_matrix[t, p] += 1
    print(confusion_matrix)
        

if __name__ == "__main__":
    seed = 7777
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    args = parseargs()
    utils.prepare_cifar10("./data")
    test(args)
