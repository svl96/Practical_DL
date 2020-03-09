import torch 
from time import time
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Accuracy(torch.nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, outputs, targets):
        _, preds = torch.max(outputs, 1)
        return torch.mean((preds == targets).double())


class DistillationAccuracy(torch.nn.Module):
    def __init__(self):
        super(DistillationAccuracy, self).__init__()

    def forward(self, outputs, targets):
        student, teacher = outputs
        _, preds = torch.max(student, 1)
        return torch.mean((preds == targets).double())


class DistillationLoss(torch.nn.Module):
    def __init__(self, tau=20, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.tau = tau
        self.alpha = alpha
        self.KLDiv_criterion = torch.nn.KLDivLoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        student_out, teacher_out = outputs

        KLDiv_loss = self.KLDiv_criterion(F.log_softmax(student_out/self.tau),
                                         F.softmax(teacher_out/self.tau)) 

        cross_entropy_loss = self.cross_entropy(student_out, targets)

        return self.alpha * KLDiv_loss + (1 - self.alpha) * cross_entropy_loss


class TimeProfiler:
    def __init__(self):
        self.timers = {}
        self.timer_step = {}

    def reset(self):
        self.timers = {}
        self.timer_step = {}

    def has_timer(self, name):
        return (name in self.timers) and (name in self.timer_step)
    
    def start_timer(self, name):
        if not self.has_timer(name):
            self.timers[name] = []
        
        self.timer_step[name] = time()

    def loop_timer(self, name):
        if not self.has_timer(name):
            self.start_timer()
            return
        
        self.timers[name].append(time() - self.timer_step[name])
        self.timer_step[name] = time()
    
        return self.timers[name][-1]
    
    def get_mean(self, name):
        if (not self.has_timer(name)) or (len(self.timers[name]) == 0):
            return 0
        
        return sum(self.timers[name]) / len(self.timers[name])


def plot_hist(hist, title, ylabel, xlabel='Epochs', filename='plot.png'):
    plt.title(title)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(hist['train'], label='train')
    plt.plot(hist['val'], label='val')
    plt.legend()
    plt.savefig(filename)
    plt.show()

def get_test_data(ids_file, annotation_file):
        
    with open(ids_file, 'r') as f:
        ids = f.readlines()
    ids_map = {val: i for i, val in enumerate(sorted([val.strip() for val in ids]))}    
    
    with open(annotation_file, 'r') as f:
        annotations = f.readlines()
    data = [row.split('\t')[:2] for row in annotations]
    
    filenames, targets = [], []
    for fn, tr in data:
        filenames.append(fn)
        targets.append(ids_map[tr])
    
    return filenames, targets
