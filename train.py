from __future__ import print_function
import os, shutil
import numpy as np
import logging
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import models
from LRSGDC import LRSGDC
from LRSGDR import LRSGDR
from options import Options


def main():
    global tb_writer, logger, logger_results
    
    opt = Options()
    opt.parse()
    opt.save_options()
    
    tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.save_dir))

    if opt.random_seed >= 0:
        torch.manual_seed(opt.random_seed)
        torch.cuda.manual_seed(opt.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.random_seed)
    else:
        torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpu)

    # set up logger
    logger, logger_results = setup_logger(opt)
    opt.print_options(logger)

    # define models and same initialization
    if opt.model_name not in models.__model_names__:
        raise NotImplementedError()
    model = models.__dict__[opt.model_name](num_classes=10)
    logger.info('Model: {:s}'.format(opt.model_name))
    model = torch.nn.DataParallel(model).cuda()

    # define optimizers
    if opt.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == 'lrsgdr':
        optimizer = LRSGDR(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay,
                           lambda_w=opt.lambda_w)
    elif opt.optimizer.lower() == 'lrsgdc':
        optimizer = LRSGDC(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay,
                           lambda_w=opt.lambda_w)
    else:
        raise NotImplementedError()

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define transformation and load dataset
    transform_train, transform_test = opt.transform_train, opt.transform_test
    trainset = torchvision.datasets.CIFAR10(root=opt.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root=opt.data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=1)

    inner_trainset = torchvision.datasets.CIFAR10(root=opt.data_dir, train=True, download=True, transform=transform_train)
    inner_data_loader = torch.utils.data.DataLoader(inner_trainset, batch_size=opt.batch_size, shuffle=True, num_workers=1)

    for epoch in range(opt.start_epoch + 1, opt.num_epoch + 1):  # 1 base
        adjust_learning_rate(optimizer, epoch, opt.lr)

        train_stats = train(opt, model, train_loader, inner_data_loader, optimizer, criterion, epoch)
        test_stats = test(opt, model, test_loader, criterion, epoch)

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, opt.save_dir)

        # save training results
        logger_results.info('{:d}\t{r1[0]:.4f}\t{r1[1]:.4f}\t{r2[0]:.4f}\t{r2[1]:.4f}'
                            .format(epoch, r1=train_stats, r2=test_stats))

        # tensorboard logs
        tb_writer.add_scalars('epoch_losses', {'train_loss': train_stats[0], 'test_loss': test_stats[0]}, epoch)
        tb_writer.add_scalars('epoch_accuracies',  {'train_acc': train_stats[1], 'test_acc': test_stats[1]}, epoch)
        
    tb_writer.close()
    for i in list(logger.handlers):
        logger.removeHandler(i)
        i.flush()
        i.close()
    for i in list(logger_results.handlers):
        logger_results.removeHandler(i)
        i.flush()
        i.close()


def train(opt,  model, train_loader, inner_data_loader, optimizer, criterion, epoch):
    # loss, acc
    train_result = AverageMeter(2)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        if batch_idx == 0:
            logger.info('lr {:.3g}'.format(optimizer.param_groups[0]['lr']))

        if opt.optimizer.lower() == 'sgd':
            # official SGD
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        else:
            output = model(data)
            loss = criterion(output, target)

            inner_num_iter = np.random.geometric(opt.geometric_p, 1)[0] if opt.optimizer.lower() == 'lrsgdr' else opt.inner_num_iter
            for i, (inner_data, inner_target) in enumerate(inner_data_loader):
                if i == inner_num_iter:
                    break

                inner_data, inner_target = inner_data.cuda(), inner_target.cuda()
                optimizer.zero_grad()
                inner_output = model(inner_data)
                inner_loss = criterion(inner_output, inner_target)
                inner_loss.backward()
                optimizer.step(i)

        pred = output.max(1, keepdim=True)[1]    # get the index of the max probability
        acc = pred.eq(target.view_as(pred)).sum().item() / float(opt.batch_size)

        result = [loss.item(), acc]
        train_result.update(result, data.size(0))

        if batch_idx % opt.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {r[0]:.4f}\tAcc: {r[1]:.4f}'
                        .format(epoch, batch_idx * len(data),  len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), r=train_result.avg))

    logger.info('=> Train Epoch: {}\tLoss: {r[0]:.4f}\tAcc: {r[1]:.4f}'.format(epoch, r=train_result.avg))
    return train_result.avg


def test(opt, model, test_loader, criterion, epoch):
    # loss, accuracy
    test_result = AverageMeter(2)

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            acc = pred.eq(target.view_as(pred)).sum().item() / float(opt.batch_size)

            result = [loss.item(), acc]
            test_result.update(result, data.size(0))

        logger.info('=> Test Epoch: {}\tLoss: {r[0]:.4f}\tAcc: {r[1]:.4f}\n'.format(epoch, r=test_result.avg))
    return test_result.avg


def adjust_learning_rate(optimizer, epoch, initial_lr):
    # step decay
    if epoch <= 150:
        lr = initial_lr
    elif epoch <= 250:
        lr = initial_lr * 0.1
    else:
        lr = initial_lr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(save_dir)
    torch.save(state, filename)


def setup_logger(opt):
    mode = 'a' if opt.checkpoint else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train.log'.format(opt.save_dir), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)

    # create logger for iteration results
    logger_iter_results = logging.getLogger('results_iteration')
    logger_iter_results.setLevel(logging.DEBUG)

    # set up logger for each result
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.save_dir), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.save_dir))
    if mode == 'w':
        logger_results.info('epoch\tTrain_loss\tTrain_acc\tTest_loss\tTest_acc')
    return logger, logger_results


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = np.zeros(self.shape)

    def update(self, val, n=1):
        val = np.array(val)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
