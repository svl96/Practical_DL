import torch
import torchvision
from torch import nn


def conv(cin, cout, kernel_size=(3,3), padding=(1,1), stride=(1,1)):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size, padding=padding, stride=stride),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
        nn.MaxPool2d((2,2))
    )


class DistillModel(nn.Module):
    def __init__(self, teacher, student, tau=1):
        super(DistillModel, self).__init__()
        self.teacher = teacher
        self.student = student
        self.tau = tau

        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def forward(self, input):
        teacher_out = self.teacher(input)
        student_out = self.student(input)

        return student_out, teacher_out.detach()


def get_distill_model(teacher_file='', tau=1, **kwargs):
    if teacher_file == '':
        return

    teacher = get_standard_model()
    teacher.load_state_dict(torch.load(teacher_file))
    student = get_small_model()

    return DistillModel(teacher, student, tau=tau)


def get_model_pretrained(**kwards):
    model = torchvision.models.resnet18(pretrained=True)
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 200)

    return model


def get_big_model(cin=3, cout=200, base=64, softmax=True):
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
        # nn.LogSoftmax(dim=1)
    )


def get_standard_model(cin=3, cout=200, base=64, softmax=False):
    return nn.Sequential(
        conv(cin, base), # 3, 64, 64 -> 64*32*32
        conv(base, base*2), # 64, 32, 32 -> 128*16*16
        nn.Dropout(0.2),
        conv(base*2, base*4), # 128, 16, 16 -> 256*8*8
        conv(base*4, base*8), # 256, 8, 8 -> 512*4*4
        nn.AdaptiveAvgPool2d(output_size=(1,1)),
        nn.Flatten(),
        nn.Linear(base*8, 200),
        # nn.LogSoftmax(dim=1)
    )


def get_small_model(cin=3, cout=200, base=64, softmax=False):
    return nn.Sequential(
        conv(cin, base), # 3, 64, 64 -> 64*32*32
        conv(base, base*2), # 64, 32, 32 -> 128*16*16
        nn.AdaptiveAvgPool2d(output_size=(1,1)),
        nn.Flatten(),
        nn.Linear(base*2, 200),
        # nn.LogSoftmax(dim=1)
    )


def get_model(name, **kwards):
    models = {
        'resnet': get_model_pretrained,
        'big': get_big_model,
        'standard': get_standard_model,
        'small': get_small_model,
        'distill': get_distill_model
    }

    return models[name](**kwards)
