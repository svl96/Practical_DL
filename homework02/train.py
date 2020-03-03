import numpy as np
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import argparse

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import ImgsDataset
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint_sequential
from utils import TimeProfiler, Accuracy


EPOCH_TIMER = 'epoch_timer'
BATCH_TIMER = 'batch_timer'



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


def train_one_epoch(model, dataloader, criterion, optimizer, metric, device, args, profiler):
    total_loss = 0
    total_acc = 0
    model.train(True)
    optimizer.zero_grad()

    profiler.start_timer(BATCH_TIMER)

    for i, (samples, targets) in enumerate(dataloader):
        samples = samples.to(device)
        targets = targets.to(device)

        if args.checkpoint > 0:
            outputs = checkpoint_sequential(model, args.checkpoint, samples)
        else:
            outputs = model(samples)
        loss = criterion(outputs, targets)
        _, preds = torch.max(outputs, 1)
        total_acc += metric(preds, targets).item()

        loss.backward()
        if args.batch_count <= 1 or (i+1) % args.batch_count == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        profiler.loop_timer(BATCH_TIMER)

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


def train_model(model, dataloaders, criterion, optimizer, metric, device, args, profiler):
    loss_hist = {'train': [], 'val': []}
    acc_hist = {'train': [], 'val': []}
    epochs = args.epochs
    profiler.start_timer(EPOCH_TIMER)

    for epoch in range(epochs):
        loss, acc = train_one_epoch(model, dataloaders['train'], criterion,
                                    optimizer, metric, device, args, profiler)
        
        val_loss, val_acc = eval_model(model, dataloaders['val'], criterion, metric, device)
        print("Epoch [{}/{}] Time: {:.2f}s; BatchTime:{:.2f}s; Loss: {:.4f}; Accuracy: {:.4f}; ValLoss: {:.4f}; ValAccuracy: {:.4f}".format(
                epoch + 1, epochs, profiler.loop_timer(EPOCH_TIMER),
                profiler.get_mean(BATCH_TIMER), loss, acc, val_loss, val_acc))
            
        loss_hist['train'].append(loss)
        loss_hist['val'].append(val_loss)
        
        acc_hist['train'].append(acc)
        acc_hist['val'].append(val_acc)
        
        if sum(acc_hist['val'][-2:])/2 > 0.4:
            break

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


def run_training(model, args, profiler):
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("device %s" %device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.opt_lr)
    metric = Accuracy()
    dataloaders = get_dataloaders(batch_size=args.batch)
    
    print('Run training')
    model, loss_hist, acc_hist = train_model(model, dataloaders, criterion,
                                             optimizer, metric, device, args, profiler)
    if args.save_file != '':
        torch.save(model.state_dict(), args.save_file)
    loss_test, acc_test = eval_model(model, dataloaders['test'], criterion, metric, device)
    print("\nTEST Loss: {:.4f}; Accuracy: {:.4f}\n".format(loss_test, acc_test))
    
    if args.plot_hist:
        plot_hist(loss_hist, "Loss History", 'CrossEntropyLoss')
        plot_hist(acc_hist, "Accuracy History", 'Accuracy')
    
    return model, loss_test, acc_test


def load_model(path):
    model = get_model()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def get_args():
    parser = argparse.ArgumentParser(description='Image Classifier training')
    parser.add_argument('--save_file', type=str, default='', help='File to save model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch', type=int, default=200, help='Size of Batch')
    parser.add_argument('--checkpoint', type=int, default=-1, help='Size of checkpoint')
    parser.add_argument('--opt_lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument("--batch_count", type=int, default=1, help='Multiplier for computational batch size equals "total=batch_count*batch"')
    parser.add_argument('--plot_hist', action='store_true', default=False, help='Plot history of training')
    parser.add_argument('--pretrained',  action='store_true', default=False, help='Train resnet18 pretrained model')
    parser.add_argument('--memory_usage',  action='store_true', default=False, help='Show memory usage')


    return parser.parse_args()

from time import sleep

def main(pretrained=True, epochs=5):
    args = get_args()
    if args.pretrained:
        print("Pretrained mode")
        model = get_model_pretrained()
    else:
        print("train model")
        model = get_model()

    profiler = TimeProfiler()

    run_training(model, args, profiler) 
    if args.memory_usage:
        print(f"Peak memory usage by Pytorch tensors: {(torch.cuda.max_memory_allocated() / 1024 / 1024):.2f} Mb")


if __name__ == '__main__':
    main()
