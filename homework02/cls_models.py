import torch
import torchvision
from torch import nn


def get_model_pretrained():
    model = torchvision.models.resnet18(pretrained=True)
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 200)

    return model


def conv(cin, cout, kernel_size=(3,3), padding=(1,1), stride=(1,1)):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size, padding=padding, stride=stride),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
        nn.MaxPool2d((2,2))
    )


def get_model(cin=3, cout=200, base=64, softmax=True):
    return nn.Sequential(
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
