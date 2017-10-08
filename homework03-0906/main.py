#!/usr/bin/env python
# encoding:utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import sampler, DataLoader

# torchvision datasets as the data
import torchvision.datasets as tvdatasets
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg') # save pics
import matplotlib.pyplot as plt
import numpy as np
import copy
plt.rcParams["figure.figsize"] = [10, 30]

# set hyper parameters
image_size = 28
num_epochs = 100
batch_size = 128

use_cuda = torch.cuda.is_available()
print("GPU Avaialble: ", use_cuda)

# set cuda tensor if cuda available
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
ltype = torch.cuda.LongTensor if use_cuda else torch.LongTensor


def datasets_classifier(train_all=False):
    datasets = tvdatasets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)
    indices_0_4 = []
    indices_5_9 = []

    for idx in range(len(datasets)):
        t = int(datasets[idx][1])
        if t in [0, 1, 2, 3, 4]:
            indices_0_4.append(idx)
        else:
            indices_5_9.append(idx)

    # subset random sampler ?
    sampler_0_4 = sampler.SubsetRandomSampler(indices_0_4)
    sampler_5_9 = sampler.SubsetRandomSampler(indices_5_9)

    # zero to four data loader
    loader_0_4 = DataLoader(dataset=datasets,
                            batch_size=batch_size,
                            sampler=sampler_0_4)

    # five to nine data loader
    loader_5_9 = DataLoader(dataset=datasets,
                            batch_size=batch_size,
                            sampler=sampler_5_9)

    # all data loader
    loader_all = DataLoader(dataset=datasets,
                            batch_size=batch_size,
                            shuffle=True)

    if train_all is False:
        return loader_0_4, loader_5_9
    else:
        return loader_0_4, loader_all


def load_test_datasets():
    val_size = 5000
    test_dataset = tvdatasets.MNIST(root='./data',
                                    train=False,
                                    transform=transforms.ToTensor())

    val_indices = range(val_size)
    test_indices = range(val_size, len(test_dataset))

    val_smapler = sampler.SubsetRandomSampler(val_indices)
    test_sampler = sampler.SubsetRandomSampler(test_indices)

    val_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            sampler=val_smapler)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             sampler=test_sampler)

    return val_loader, test_loader


cnn_depth = [8, 16]
fc = [1024, 128]  # Fully connected layer


class ConvNet1(nn.Module):
    def __init__(self):
        super(ConvNet1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=cnn_depth[0],
                               kernel_size=5,
                               stride=1,
                               padding=2)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=cnn_depth[0],
                               out_channels=cnn_depth[1],
                               kernel_size=5,
                               stride=1,
                               padding=2)

        self.fc1 = nn.Linear(in_features=image_size // 4 * image_size // 4 * cnn_depth[1],
                             out_features=fc[0],
                             bias=True)

        self.fc2 = nn.Linear(in_features=fc[0],
                             out_features=fc[1],
                             bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, image_size // 4 * image_size // 4 * cnn_depth[1])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x)
        return x


class ConvNet2(ConvNet1):
    def __init__(self):
        super(ConvNet2, self).__init__()

    def set_filter_values(self, net):
        self.conv1.weight.data = copy.deepcopy(net.conv1.weight.data)
        self.conv1.bias.data = copy.deepcopy(net.conv1.bias.data)
        self.conv2.weight.data = copy.deepcopy(net.conv2.weight.data)
        self.conv2.bias.data = copy.deepcopy(net.conv2.bias.data)

        self.conv1 = self.conv1.cuda() if use_cuda else self.conv1
        self.conv2 = self.conv2.cuda() if use_cuda else self.conv2


def rightness(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


def train_net(net, num_exps, num_epochs, data_loader, val_loader):
    """
    return:
        net: network after training
        records: a list including train_loss, val_loss, train_rate & val_rate
        records = [[train_loss, val_loss, train_rate, val_rate],...]
    """

    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # 定义优化器
    records = []

    for epoch in range(num_epochs):
        train_rights, train_losses = [], []
        # 先用0～4这5个数字训练卷积网络ConvNet
        for idx, (data, target) in enumerate(data_loader):
            data, target = Variable(data), Variable(target)
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            net.train() #给网络做训练标记，
            output = net(data) # 完成一次预测
            loss = criterion(output, target) # 计算误差
            optimizer.zero_grad() # 清空梯度
            loss.backward() # 反向传播
            optimizer.step() # 一步随机梯度下降
            right = rightness(output, target) # 计算准确率，返回正确数值 （正确样例数，总样本数）
            train_rights.append(right)
            loss = loss.cpu() if use_cuda else loss
            train_losses.append(loss.data.numpy())

            if idx % 100 == 0:
                train_r = (sum([_[0] for _ in train_rights]),
                           sum([_[1] for _ in train_rights]))
                net.eval() # 测试标记
                val_rights, val_losses = [], []
                for (data, target) in val_loader:
                    data, target = Variable(data), Variable(target)
                    if use_cuda:
                        data, target = data.cuda(), target.cuda()
                    output = net(data)
                    loss = criterion(output, target) # 计算误差
                    optimizer.zero_grad() # 清空梯度
                    loss.backward() # 反向传播
                    optimizer.step() # 一步随机梯度下降
                    right = rightness(output, target)
                    val_rights.append(right)
                    loss = loss.cpu() if use_cuda else loss
                    val_losses.append(loss.data.numpy())

                val_r = (sum([_[0] for _ in val_rights]),
                         sum([_[1] for _ in val_rights]))

                train_acc = 100. * train_r[0] / train_r[1]
                val_acc = 100. * val_r[0] / val_r[1]
                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)
                print("# of exp: {}, # of epoch：{}, [{}/{} {:.0f}%]\t \
                      train loss :{:.2f}\t \
                      val loss:{:.2f}\t \
                      train acc:{:.2f}%\t \
                      val acc: {:.2f}%"
                      .format(num_exps + 1, epoch, idx, len(data_loader), 100. * idx / len(data_loader),
                              train_loss,
                              val_loss,
                              train_acc,
                              val_acc))

                records.append([train_loss, val_loss, train_acc, val_acc])

    return net, records


def save_net(net, tag):
    import datetime
    import os
    save_dir = './models'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_file = 'ConvNet_{}.{}'.format(tag, datetime.datetime.now().strftime('%y%m%d-%H%M%S'))
    save_path = os.path.join(os.getcwd(), save_dir, save_file)
    torch.save(net.state_dict(), save_path)
    print('Save model OK: ', save_path)


def save_results_pics(exp, records_dict):
    import os
    save_dir = 'pics'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for tag, records in records_dict.items():
        train_loss = [rcd[0] for rcd in records[0]]
        val_loss = [rcd[1] for rcd in records[0]]

        plt.plot(np.arange(len(records[0])), train_loss, label='train_loss_' + str(tag))
        plt.plot(np.arange(len(records[0])), val_loss, label='val_loss_' + str(tag))

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.savefig('./pics/train_and_val_loss_' + exp + '.jpg')
    plt.close('all')

    train_loss = np.zeros(len(train_loss))
    val_loss = np.zeros(len(val_loss))
    for tag, records in records_dict.items():
        train_loss += np.array([rcd[0] for rcd in records[0]])
        val_loss += np.array([rcd[1] for rcd in records[0]])

    plt.plot(np.arange(len(records[0])), train_loss / 5, label='train_loss_ave')
    plt.plot(np.arange(len(records[0])), val_loss / 5, label='val_loss_ave')

    plt.xlabel('epoch')
    plt.ylabel('ave_loss')
    plt.legend()

    plt.savefig('./pics/ave_train_and_val_loss_' + exp + '.jpg')
    plt.close('all')

    for tag, records in records_dict.items():
        train_acc = [rcd[2] for rcd in records[0]]
        val_acc = [rcd[3] for rcd in records[0]]
        test_acc = [records[1]]
        plt.plot(np.arange(len(records[0])), val_acc, label='val_acc_' + str(tag))
        plt.plot(np.arange(len(records[0])), train_acc, label='train_acc_' + str(tag))
        plt.plot(np.arange(len(records[0])), test_acc * len(records[0]), label='test_acc_' + str(tag))

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('./pics/train_test_val_accuracy_' + exp + '.jpg')
    plt.close('all')

    train_acc = np.zeros(len(train_acc))
    val_acc = np.zeros(len(val_acc))
    test_acc = np.zeros(len(test_acc))
    for tag, records in records_dict.items():
        train_acc += np.array([rcd[2] for rcd in records[0]])
        val_acc += np.array([rcd[3] for rcd in records[0]])
        test_acc += np.array([records[1]])

    plt.plot(np.arange(len(records[0])), val_acc / 5, label='val_acc_ave')
    plt.plot(np.arange(len(records[0])), train_acc / 5, label='train_acc_ave')
    plt.plot(np.arange(len(records[0])), [test_acc / 5] * len(records[0]), label='test_acc_ave')

    plt.xlabel('epoch')
    plt.ylabel('ave_accuracy')
    plt.legend()
    plt.savefig('./pics/ave_train_test_val_accuracy_' + exp + '.jpg')
    plt.close('all')

def test_net(net, test_loader):
    vals = []
    net.eval()
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = net(data)
        val = rightness(output, target)
        vals.append(val)

    rights = (sum([tup[0] for tup in vals]),
              sum([tup[1] for tup in vals]),)

    test_acc = 100.0 * rights[0] / rights[1]
    return test_acc


def experiment_1(exp_epoch):
    tag = "1st exp at", exp_epoch
    loader_0_4, loader_5_9 = datasets_classifier()
    val_loader, test_loader = load_test_datasets()

    net1 = ConvNet1()
    net1 = net1.cuda() if use_cuda else net1
    net1, _ = train_net(net1, exp_epoch, num_epochs, loader_0_4, val_loader)

    net2 = ConvNet2()
    net2.set_filter_values(net1)
    net2 = net2.cuda() if use_cuda else net2
    net2, records = train_net(net2, exp_epoch, num_epochs, loader_5_9, val_loader)

    save_net(net2, tag)
    # save_results_pics(records, tag)

    test_acc = test_net(net2, test_loader)
    print(tag, "Acc on test dataset: ", test_acc, "%")
    return (tag, records, test_acc)


def experiment_2(exp_epoch):
    tag = "2nd exp at ", exp_epoch
    loader_0_4, loader_all = datasets_classifier(train_all=True)
    val_loader, test_loader = load_test_datasets()

    net1 = ConvNet1()
    net1 = net1.cuda() if use_cuda else net1
    net1, _ = train_net(net1, exp_epoch, num_epochs, loader_0_4, val_loader)

    net2 = ConvNet2()
    net2.set_filter_values(net1)
    net2 = net2.cuda() if use_cuda else net2
    net2, records = train_net(net2, exp_epoch, num_epochs, loader_all, val_loader)

    save_net(net2, tag)

    test_acc = test_net(net2, test_loader)
    print(tag, "Acc on test dataset: ", test_acc, "%")
    return (tag, records, test_acc)


def experiment_3(exp_epoch):
    tag = "3rd exp at ", exp_epoch
    _, loader_all = datasets_classifier(train_all=True)
    val_loader, test_loader = load_test_datasets()

    net1 = ConvNet1()
    net1 = net1.cuda() if use_cuda else net1
    net1, records = train_net(net1, exp_epoch, num_epochs, loader_all, val_loader)

    save_net(net1, tag)

    test_acc = test_net(net1, test_loader)
    print(tag, "Acc on test dataset: ", test_acc, "%")
    return (tag, records, test_acc)


num_experiment_repeat = 5


def main():
    print('========= transfer learning exp 1 ===========')
    exp_1_records = {}
    for exp_epoch in range(num_experiment_repeat):
        tags, records, test_acc = experiment_1(exp_epoch)
        exp_1_records[tags] = (records, test_acc)
    save_results_pics('exp_1', exp_1_records)

    print('========= transfer learning exp 2 ===========')
    exp_2_records = {}
    for exp_epoch in range(num_experiment_repeat):
        tags, records, test_acc = experiment_2(exp_epoch)
        exp_2_records[tags] = (records, test_acc)
    save_results_pics('exp_2', exp_2_records)

    print('========= transfer learning exp 3 ===========')
    exp_3_records = {}
    for exper_time in range(num_experiment_repeat):
        tags, records, test_acc = experiment_3(exper_time)
        exp_3_records[tags] = (records, test_acc)
    save_results_pics('exp_3', exp_3_records)


main()
