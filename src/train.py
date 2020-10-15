import argparse
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torchvision import transforms
from torch.utils.data import DataLoader
from model import Classifier

device = "cuda" if torch.cuda.is_available() else "cpu"
cifar10_path = "data/cifar-10-batches-py"

def parseargs():
    parser = argparse.ArgumentParser(description="Classifier trainer")
    parser.add_argument("--epoch", "-e", type=int, default=100)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--weight", "-w", type=str, default="")
    parser.add_argument("--alpha", "-a", type=float, default=1.0)
    return parser.parse_args()

def learning_rate_schedule(epoch):
    if epoch < 25:
        return 1
    elif epoch < 50:
        return 1. / 2**1
    elif epoch < 75:
        return 1. / 2**2
    elif epoch < 100:
        return 1. / 2**3
    elif epoch < 125:
        return 1. / 2**4
    elif epoch < 150:
        return 1. / 2**5
    elif epoch < 175:
        return 1. / 2**6
    else:
        return 1. / 2**7


def train(args):
    num_epoch = args.epoch
    lr = args.learning_rate
    batch_size = args.batch_size
    weight = args.weight
    alpha = args.alpha

    # preprocessing
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Pad(padding=4, padding_mode="reflect"),
            transforms.RandomCrop((32, 32)),
            transforms.ColorJitter(),
            #transforms.RandomRotation(degrees=20),
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomErasing(), 
        ]
    )

    # load training dataset
    X_train, y_train, _, _ = utils.load_cifar10(cifar10_path)

    assert X_train.shape[0] == y_train.shape[0]
    print(f"Dataset size (train) = {y_train.shape[0]}")

    train_data = utils.MyDataset(x=X_train, y=y_train, transform=transform)

    # weighted sampling
    weighted_sampler = utils.set_weighted_sampler(label_data=y_train)
    train_loader = DataLoader(train_data, batch_size, shuffle=False, sampler=weighted_sampler, num_workers=2)

    # normal sampling
    #train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=2)

    # model, optimizer, loss function
    model = Classifier().to(device)
    if len(weight) > 0:
        model.load_state_dict(torch.load(weight))
   
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = learning_rate_schedule)
    
    # loss function
    cross_entropy_loss = nn.CrossEntropyLoss().to(device)
    mse_loss = nn.MSELoss().to(device)

    # train
    train_loss = []
    for epoch in range(num_epoch):
        train_tmp_loss = []
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward
            outputs, autoencoder_outputs = model(inputs)

            # calc categorical cross entropy
            loss1 = cross_entropy_loss(outputs, labels)
            # calc mean squared loss
            loss2 = mse_loss(inputs, autoencoder_outputs)

            loss = loss1 + alpha * loss2

            # backpropagate losses
            loss.backward()
            train_tmp_loss.append(loss.item() * len(inputs))
            optimizer.step()

            if i % 100 == 99:
                print(f"epoch: {epoch+1}, batch: {i+1}, loss: {loss.item():.3f}")

        train_loss.append(sum(train_tmp_loss) / y_train.shape[0])

        # adjust learning rate according to the schedule
        scheduler.step()
    
        # save parameters
        weight_path = pathlib.Path(f"weight/epoch_{epoch+1:03d}_{alpha}.pth")
        torch.save(model.state_dict(), weight_path)

        print(train_loss)

if __name__ == "__main__":
    args = parseargs()
    utils.prepare_cifar10("./data")
    train(args)
