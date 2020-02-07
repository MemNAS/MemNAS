import argparse
import os
import shutil
import time
import util
from torch.utils.serialization import load_lua
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
import numpy as np
import importlib
import sys
from torch.optim import lr_scheduler
import math
from torchvision import datasets, transforms
from util.misc import CSVLogger
from util.cutout import Cutout
from tqdm import tqdm
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model_names.append('mobilenet')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--step', default=70, type=int, metavar='N',help='step')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=True, type=str, metavar='./model_best.pth.tar',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--tau', dest='tau', default=3.0, type=float, help='tau')
parser.add_argument('--lamda', dest='lamda', default=0.8, type=float, help='lamda')
parser.add_argument('--network', dest='network', help='network')
parser.add_argument('--con', dest='con', type=int, help='con')
parser.add_argument('--net', dest='net', type=str, help='net')
parser.add_argument('--cot_start', dest='cot_start', type=int, help='net')
parser.add_argument('--cot_end', dest='cot_end', type=int, help='net')
parser.add_argument('--pre_net', dest='pre_net', type=str, help='net')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
torch.cuda.set_device(0)

best_prec1 = 0
gpu = 0


preprocess = torchvision.transforms.Compose([
			    torchvision.transforms.RandomCrop(32,padding=4),
			    torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.ToTensor(),
			    torchvision.transforms.Normalize( (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])
preprocess_v = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
			    torchvision.transforms.Normalize( (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])


def main():
    global args
    args = parser.parse_args()
    train_transform = transforms.Compose([])
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    #    train_transform.transforms.append(transforms.ColorJitter(
    #            brightness=0.1*torch.randn(1),
    #            contrast=0.1*torch.randn(1),
    #            saturation=0.1*torch.randn(1),
    #            hue=0.1*torch.randn(1)))
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])
    train_dataset = datasets.CIFAR10(root='cifar/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='cifar/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)

    # create model
    #
    #for net_count in range(args.cot_start,args.cot_end):
        #import networks/round1/1
    net_count=0
    best_prec1=0

    net_file_path="networks/"
    sys.path.append(net_file_path)
    net_temp=importlib.import_module("testnet")
    net=getattr(net_temp,"testnet")

    model_m = net(32)
    model_dict = model_m.state_dict()

    '''
    pretrain_model_path="model/round5_Score/"+"net242_1"+"_params.pkl"

    pretrained_dict=torch.load(pretrain_model_path)

    for k in model_dict.keys():
        if "module."+k in pretrained_dict.keys():
            if model_dict[k].size()==pretrained_dict["module."+k].size():
                model_dict[k]=pretrained_dict["module."+k]
            if len(pretrained_dict["module."+k].size())==2:
                if (model_dict[k].size()[1])>pretrained_dict["module."+k].size()[1]:
                    max1=pretrained_dict["module."+k].size()[1]
                else:
                    max1=(model_dict[k].size()[1])

                model_dict[k][:,0:max1]=pretrained_dict["module."+k][:,0:max1]
            if len(pretrained_dict["module."+k].size())==4:
                if (model_dict[k].size()[0])>pretrained_dict["module."+k].size()[0]:
                    max0=pretrained_dict["module."+k].size()[0]
                else:
                    max0=(model_dict[k].size()[0])
                if (model_dict[k].size()[1])>pretrained_dict["module."+k].size()[1]:
                    max1=pretrained_dict["module."+k].size()[1]
                else:
                    max1=(model_dict[k].size()[1])
                print model_dict[k].size()
                print pretrained_dict["module."+k].size()
                model_dict[k][0:max0,0:max1,:,:]=pretrained_dict["module."+k][0:max0,0:max1,:,:]

    model_m.load_state_dict(model_dict)
    '''







    #net_file_path="networks/round1/"
    #sys.path.append(net_file_path)
    '''
    net=getattr(test,args.network)
    model_m = net(args.con)
    '''


    print model_m

    model_m = torch.nn.DataParallel(model_m).cuda()
    #model_param_path='model/mb6_fc1024/'+args.network+'_params'+'pkl'
    #model_m.load_state_dict(torch.load(model_param_path))
    #print model_m
    optimizer_m =torch.optim.SGD(model_m.parameters(), args.lr, args.momentum)
    #optimizer_m =torch.optim.Adam(model_m.parameters(), args.lr, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_m, milestones=[60,180,240], gamma=0.1)
    #model_m.load_state_dict(torch.load('model/new/start_params.pkl'))
    criterionCE = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True
    #train_loader = load_lua('cifar10.t7')
    #val_loader =train_loader.val
    #train_loader = train_loader.train

    best_acc=0

    lamda = args.lamda

    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer_m, epoch,args.epochs, 0 )
        exp_lr_scheduler.step()

        # train for one epoch
        validate(test_loader, model_m, criterionCE,best_acc)
        train(train_loader,model_m, criterionCE, optimizer_m,epoch, lamda)

        # evaluate on validation set
        prec1 = validate(test_loader, model_m, criterionCE,best_acc)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print(best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict1': model_m.state_dict(),
            #'state_dict2': model2.state_dict(),
            'best_prec1': best_prec1,
            #'optimizer1' : optimizer_m.state_dict(),
            #'optimizer2': optimizer2.state_dict()
        }, is_best,model_m,net_count)


    #print("The Best accuracy for"+"net"+str(net_count)+"is:  "+str(best_prec1))


def train(train_loader,model_m, criterionCE,optimizer_m,epoch, lamda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    # switch to train mode


    model_m.train()
    model_m=model_m.cuda(gpu)

    for param in model_m.parameters():
        param.requires_grad = True
    end = time.time()

    #input_data=train_loader.data
    #target_loader=train_loader.labels

    #for i in xrange((len(input_data))/args.batch_size):
    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        '''
        input=input_data[i*args.batch_size:(i+1)*args.batch_size]
        target=target_loader[i*args.batch_size:(i+1)*args.batch_size]

        input_batch=input.float()
        input_batch=input_batch.numpy()
        input_batch=input_batch.transpose(0, 2, 3, 1)
        input_batch = torch.stack([preprocess(Image.fromarray(np.uint8(item))) for item in input_batch])

        inputs=input_batch.float()
        inputs=inputs.view(-1,3,32,32)

        # measure data loading time
        data_time.update(time.time() - end)

        target=target.long()
        target = target.cuda(async=True)
        '''
        images = images.cuda()
        labels = labels.cuda()
        input_var = torch.autograd.Variable(images)
        target_var = torch.autograd.Variable(labels)
        #for m in model_m.modules():
        #    if isinstance(m, nn.Conv2d):
        #        print m.weight


        #mulbit_op.quaternary()
        #mulbit_op.binarization()

    #    os.exit()


        outputm = model_m(input_var)
        output=outputm

        #lossCE = criterionCE(output, target_var-1)
        lossCE = criterionCE(output, target_var)
        loss=lossCE

        #prec1, prec5 = accuracy(output.data, (target_var-1).data, topk=(1, 5))
        prec1, prec5 = accuracy(output.data, (target_var).data, topk=(1, 5))

        top1.update(prec1[0], images.size(0))
        top5.update(prec5[0], images.size(0))



        optimizer_m.zero_grad()

        loss.backward()
        #mulbit_op.restore()
        optimizer_m.step()

        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.data[0], images.size(0))
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, (len(progress_bar))/args.batch_size, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model_m, criterionCE,best_acc):
    #batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    topm = AverageMeter()
    tops1 = AverageMeter()

    # switch to evaluate mode
    model_m.eval()



    #input_data=val_loader.data
    #target_loader=val_loader.labels
    #batch_size=100
    #mulbit_op.quaternary()
    #mulbit_op.binarization()

    #for i in xrange((len(input_data))/batch_size):
    for images, labels in val_loader:
        '''
        input_batch=input_data[i*batch_size:(i+1)*batch_size]
        target=target_loader[i*batch_size:(i+1)*batch_size]
        input_batch=input_batch.float()
        input_batch=input_batch.numpy()
        input_batch=input_batch.transpose(0, 2, 3, 1)
        input_batch = torch.stack([preprocess_v(Image.fromarray(np.uint8(item))) for item in input_batch])

        inputs=input_batch.float()
        inputs=inputs.view(-1,3,32,32)

        target=target.long()



        target = target.cuda(async=True)
        inputs = inputs.cuda(async=True)
        '''
        images = images.cuda()
        labels = labels.cuda()
        input_var = torch.autograd.Variable(images)
        target_var = torch.autograd.Variable(labels)

        output=model_m(input_var)
        #loss = criterionCE(output, target_var)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, (target_var).data, topk=(1, 5))
        top1.update(prec1[0], images.size(0))
        top5.update(prec5[0], images.size(0))
    #mulbit_op.restore()


    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))


    return top1.avg


def save_checkpoint(state,is_best,model,net_count):
    #torch.save(state, filename)

    if is_best:
        print("1")

        #mulbit_op.binarization()
        if not os.path.exists('model/test/'):
            os.mkdir('model/test/')

        model_path='model/test/'+"ttt_f"+'.pkl'
        model_parameter_path='model/test/'+"tttt_f"+'_params'+'.pkl'

        torch.save(model,model_path)
        torch.save(model.state_dict(),model_parameter_path)
        #mulbit_op.restore()




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer_m, epoch,T_max, eta_min):

    #cosine lr decay

    lr=eta_min + (args.lr - eta_min) *(1 + math.cos(math.pi * epoch / T_max))/float(2)


    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // args.step))
    for param_group in optimizer_m.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
