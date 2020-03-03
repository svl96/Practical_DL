import numpy as np
from time import time
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from dataset import ImgsDataset
import matplotlib.pyplot as plt
import argparse


class TimeProfiler:
    def __init__(self):
        self.timers = {}
        self.timer_step = {}

    def reset(self):
        self.timers = {}
        self.timer_step = {}
    
    def add_timer(self, name):
        self.timers[name] = []
        self.timer_step[name] = -1

    def has_timer(self, name):
        return name in self.timers and name in self.timer_step
    
    def start_timer(self, name):
        if not self.has_timer(name):
            self.add_timer(name)
        
        self.timer_step[name] = time()

    def loop_timer(self, name):
        if not self.has_timer(name):
            self.start_timer()
            return
        
        self.timers[name].append(time() - self.timer_step[name])
        self.timer_step[name] = time()
    
        return self.timers[name]
    
    def get_mean(self, name):
        if not self.has_timer(name) or len(self.timers[name] == 0):
            return 0
        
        return sum(self.timers[name]) / len(self.timers[name])


class Accuracy(torch.nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, outputs, targets):
        return torch.mean((outputs == targets).double())


def get_model_pretrained():
    model = torchvision.models.resnet18(pretrained=True)
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 200)

    return model


def conv(cin, cout, kernel_size=(3,3), padding=(1,1), stride=(1,1)):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size, padding=padding, stride=stride),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
        nn.MaxPool2d((2,2))
    )


def get_model(cin=3, cout=200):
    base = 64
    return torch.nn.Sequential(
        conv(cin, base), # 3, 64, 64 -> 64*32*32
        conv(base, base*2), # 64, 32, 32 -> 128*16*16
        nn.Dropout(0.2),
        conv(base*2, base*4), # 128, 16, 16 -> 256*8*8
        conv(base*4, base*8), # 256, 8, 8 -> 512*4*4
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(512*4*4, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 200),
        nn.LogSoftmax(dim=1)
    )


def train_one_epoch(model, dataloader, criterion, optimizer, metric, device):
    total_loss = 0
    total_acc = 0
    model.train(True)
    checkpoint_size = 6
    optimizer.zero_grad()

    for i, (samples, targets) in enumerate(dataloader):
        samples = samples.to(device)
        targets = targets.to(device)


        if checkpoint_size > 0:
            outputs = torch.utils.checkpoint.checkpoint_sequential(model, checkpoint_size, samples)
        else:
            outputs = model(samples)
        loss = criterion(outputs, targets)
        _, preds = torch.max(outputs, 1)
        total_acc += metric(preds, targets).item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    i += 1
    total_loss /= i
    total_acc /= i

    return total_loss, total_acc


@torch.no_grad()
def eval_model(model, dataloader, criterion, metric, device):
    total_loss = 0
    total_acc = 0
    model.eval()

    for i, (samples, targets) in enumerate(dataloader):
        samples = samples.to(device)
        targets = targets.to(device)

        outputs = model(samples)
        total_loss += criterion(outputs, targets).item()
        _, preds = torch.max(outputs, 1)
        total_acc += metric(preds, targets).item()

    i += 1
    total_loss /= i
    total_acc /= i

    return total_loss, total_acc


def train_model(model, dataloaders, criterion, optimizer, metric, device, epochs=10):
    loss_hist = {'train': [], 'val': []}
    acc_hist = {'train': [], 'val': []}
    
    start = time()
    for epoch in range(epochs):
        loss, acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, metric, device)
        val_loss, val_acc = eval_model(model, dataloaders['val'], criterion, metric, device)
        print("Epoch [{}/{}] Time: {:.2f}s; Loss: {:.4f}; Accuracy: {:.4f}; ValLoss: {:.4f}; ValAccuracy: {:.4f}".format(
              epoch + 1, epochs, time() - start, loss, acc, val_loss, val_acc))
        loss_hist['train'].append(loss)
        loss_hist['val'].append(val_loss)
        
        acc_hist['train'].append(acc)
        acc_hist['val'].append(val_acc)
        
        if sum(acc_hist['val'][-2:])/2 > 0.4:
            break
        start = time()

    return model, loss_hist, acc_hist


def get_test_dataset(transform=None, 
                     base_img_dir='tiny-imagenet-200/val/images',
                     ids_file = 'tiny-imagenet-200/wnids.txt', 
                     annotation_file = 'tiny-imagenet-200/val/val_annotations.txt'):
    
    transform = transform or transforms.ToTensor()
    
    with open(ids_file, 'r') as f:
        ids = f.readlines()
    ids_map = {val: i for i, val in enumerate(sorted([val.strip() for val in ids]))}    
    
    df = pd.read_csv(annotation_file, sep='\t', header=None)    
    
    filenames, targets = [], []
    for i, (fn, tr) in df[[0, 1]].iterrows():
        filenames.append(fn)
        targets.append(ids_map[tr])
        
    return ImgsDataset(filenames, targets=targets, base_dir=base_img_dir, transform=transform)


def get_dataloaders(batch_size=200, val_size=0.2):
    print("Batch size %d" %batch_size)
    transform = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.90, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]), 

        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    }
    
    dataset = torchvision.datasets.ImageFolder('tiny-imagenet-200/train', transform=transform['train'])
    total_size = len(dataset)
    val_size = int(total_size * val_size if val_size < 1 else val_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    test_dataset = get_test_dataset(transform=transform['test'])
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  num_workers=4),
    }
    
    return dataloaders


def plot_hist(hist, title, ylabel):
    plt.title(title)
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.plot(hist['train'], label='train')
    plt.plot(hist['val'], label='val')
    plt.legend()
    plt.show()


def run_training(model, epochs=15, batch_size=200):
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("device %s" %device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(optimizer)
    metric = Accuracy()
    dataloaders = get_dataloaders(batch_size=batch_size)
    
    print('Run training')
    model, loss_hist, acc_hist = train_model(model, dataloaders, criterion, optimizer, metric, device, epochs=epochs)
    # torch.save(model.state_dict(), 'models/model_simple_mod.pth')
    loss_test, acc_test = eval_model(model, dataloaders['test'], criterion, metric, device)
    print("\nTEST Loss: {:.4f}; Accuracy: {:.4f}\n".format(loss_test, acc_test))
    
    # plot_hist(loss_hist, "Loss History", 'CrossEntropyLoss')
    # plot_hist(acc_hist, "Accuracy History", 'Accuracy')
    
    return model, loss_test, acc_test


def load_model(path):
    model = get_model()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def get_args():
    parser = argparse.ArgumentParser(description='Image Classifier training')
    # parser.add_argument('resfile', type=str, help='Result file to save')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch', type=int, default=200, help='Size of Batch')
    parser.add_argument('--checkpoint', type=int, default=-1, help='Size of checkpoint')
    parser.add_argument("--batch_mult", type=int, default=1, help='Multiplier for computational batch size equals "total=batch_mult*batch"')
    
    return parser.parse_args()

from time import sleep

def main(pretrained=False, epochs=5):
    args = get_args()
    if pretrained:
        print("Pretrained mode")
        model = get_model_pretrained()
    else:
        print("train model")
        model = get_model()
    run_training(model, epochs=args.epochs, batch_size=args.batch) 

    print(f"Peak memory usage by Pytorch tensors: {(torch.cuda.max_memory_allocated() / 1024 / 1024):.2f} Mb")


if __name__ == '__main__':
    main()
