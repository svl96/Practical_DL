import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import ImgsDataset
from torch.utils.checkpoint import checkpoint_sequential

import argparse
from utils import TimeProfiler, Accuracy, plot_hist, get_test_data
from cls_models import get_model
import logging 
from datetime import datetime


EPOCH_TIMER = 'epoch_timer'
BATCH_TIMER = 'batch_timer'
TRAIN_IMG_DIR = 'tiny-imagenet-200/train'
VAL_IMG_DIR = 'tiny-imagenet-200/val/images'
IDS_FILE = 'tiny-imagenet-200/wnids.txt'
ANNOTATION_FILE = 'tiny-imagenet-200/val/val_annotations.txt'

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
profiler = TimeProfiler()
ts = datetime.now().strftime("%Y-%m-%d %H.%M.%S")


def train_one_epoch(model, dataloader, criterion, optimizer, metric, args):
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
        total_acc += metric(outputs, targets).item()

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
def eval_model(model, dataloader, criterion, metric):
    total_loss = 0
    total_acc = 0
    model.eval()

    for i, (samples, targets) in enumerate(dataloader):
        samples = samples.to(device)
        targets = targets.to(device)

        outputs = model(samples)
        total_loss += criterion(outputs, targets).item()
        total_acc += metric(outputs, targets).item()

    i += 1
    total_loss /= i
    total_acc /= i

    return total_loss, total_acc


def train_model(model, dataloaders, criterion, optimizer, metric, args):
    loss_hist = {'train': [], 'val': []}
    acc_hist = {'train': [], 'val': []}
    epochs = args.epochs
    profiler.start_timer(EPOCH_TIMER)

    min_val_acc = 100

    for epoch in range(epochs):
        loss, acc = train_one_epoch(model, dataloaders['train'], criterion,
                                    optimizer, metric, args)
        
        val_loss, val_acc = eval_model(model, dataloaders['val'], criterion, metric)
        logging.info("Epoch [{}/{}] Time: {:.2f}s; BatchTime:{:.2f}s; Loss: {:.4f}; Accuracy: {:.4f}; ValLoss: {:.4f}; ValAccuracy: {:.4f}".format(
                epoch + 1, epochs, profiler.loop_timer(EPOCH_TIMER),
                profiler.get_mean(BATCH_TIMER), loss, acc, val_loss, val_acc))
            
        loss_hist['train'].append(loss)
        loss_hist['val'].append(val_loss)
        
        acc_hist['train'].append(acc)
        acc_hist['val'].append(val_acc)

        if val_acc < min_val_acc:
            min_val_acc = val_acc
            if args.out and args.save_best:
                torch.save(model.state_dict(), args.out)
        
        if sum(acc_hist['val'][-2:])/2 > args.stop_accuracy:
            break

    return model, loss_hist, acc_hist


def get_test_dataset(transform=None, base_img_dir=VAL_IMG_DIR,
                     ids_file = IDS_FILE, annotation_file = ANNOTATION_FILE):
    
    transform = transform or transforms.ToTensor()
    filenames, targets = get_test_data(ids_file, annotation_file)
        
    return ImgsDataset(filenames, targets=targets, base_dir=base_img_dir, transform=transform)


def get_dataloaders(batch_size=200, val_size=0.2, num_workers=4):
    logging.info("Batch size %d" %batch_size)
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
    
    dataset = torchvision.datasets.ImageFolder(TRAIN_IMG_DIR, transform=transform['train'])
    total_size = len(dataset)
    val_size = int(total_size * val_size if val_size < 1 else val_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    test_dataset = get_test_dataset(transform=transform['test'])
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  num_workers=num_workers),
    }
    
    return dataloaders


def run_training(model, args):
    logging.info("device %s" %device)
    profiler.reset()
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.opt_lr)
    metric = Accuracy()
    dataloaders = get_dataloaders(batch_size=args.batch, num_workers=args.num_workers)
    
    logging.info('Run training')
    model, loss_hist, acc_hist = train_model(model, dataloaders, criterion,
                                             optimizer, metric, args)
    # if args.out != '':
    #     torch.save(model.state_dict(), args.out)
    loss_test, acc_test = eval_model(model, dataloaders['test'], criterion, metric)
    logging.info("\nTEST Loss: {:.4f}; Accuracy: {:.4f}\n".format(loss_test, acc_test))
    
    if args.plot_hist:
        plot_hist(loss_hist, "Loss History", 'CrossEntropyLoss', filename='plots/loss_hist_{}.png'.format(ts))
        plot_hist(acc_hist, "Accuracy History", 'Accuracy', filename='plots/acc_hist_{}.png'.format(ts))
    
    return model, loss_test, acc_test


def load_model(path):
    model = get_model()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def get_args():
    parser = argparse.ArgumentParser(description='Image Classifier training')
    parser.add_argument('--model', type=str, default='standard', help='model name to train, resnet/big/standard/distill')
    parser.add_argument('--out', type=str, default='', help='File to save model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch', type=int, default=200, help='Size of Batch')
    parser.add_argument('--checkpoint', type=int, default=-1, help='Size of checkpoint')
    parser.add_argument('--opt_lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument("--batch_count", type=int, default=1, help='Multiplier for computational batch size equals "total=batch_count*batch"')
    parser.add_argument('--plot_hist', action='store_true', default=False, help='Plot history of training')
    parser.add_argument('--pretrained',  action='store_true', default=False, help='Train resnet18 pretrained model')
    parser.add_argument('--memory_usage',  action='store_true', default=False, help='Show memory usage')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--stop_accuracy', type=float, default=0.4, help='Accuracy for early stopping')
    parser.add_argument('--save_best', action='store_false', default=True, help='Save model with better result')
    parser.add_argument('--teacher_file', type=str, default='', help='state dict for teacher model for distill')

    return parser.parse_args()


def main():
    # timestamp = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    logfile = "train_logs/log_%s.txt" % ts
    logging.basicConfig(filename=logfile, level=logging.INFO, format=u'%(message)s')
    logging.info("Logging Timestamp %s\n" % ts)
    args = get_args()
    profiler.reset()
    if args.out == '':
        args.out = 'saved_models/model_%s_%s.txt' % (args.model, ts)
    logging.info(args)

    model = get_model(args.model, teacher_file=args.teacher_file) if args.model=='distill' else get_model(args.model)

    run_training(model, args) 
    if args.memory_usage:
        logging.info(f"Peak memory usage by Pytorch tensors: {(torch.cuda.max_memory_allocated() / 1024 / 1024):.2f} Mb")


if __name__ == '__main__':
    main()
