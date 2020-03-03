import torch 
from time import time
import matplotlib.pyplot as plt


class Accuracy(torch.nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, outputs, targets):
        return torch.mean((outputs == targets).double())


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
        return (name in self.timers) and (name in self.timer_step)
    
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
    
        return self.timers[name][-1]
    
    def get_mean(self, name):
        if not self.has_timer(name) or len(self.timers[name]) == 0:
            return 0
        
        return sum(self.timers[name]) / len(self.timers[name])


def plot_hist(hist, title, ylabel):
    plt.title(title)
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.plot(hist['train'], label='train')
    plt.plot(hist['val'], label='val')
    plt.legend()
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
