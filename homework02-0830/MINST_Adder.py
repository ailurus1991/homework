# coding: utf-8

import arrow
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

# get_ipython().magic('matplotlib inline')


image_size = 28  #图像的总尺寸28*28
num_classes = 10  #标签的种类数
num_epochs = 100  #训练的总循环周期
batch_size = 64


train_dataset = dsets.MNIST(root='./data',  #文件存放路径
                            train=True,   #提取训练集
                            transform=transforms.ToTensor(),  #将图像转化为Tensor
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# 由于每一个样本需要输入两个图片，因此每一个loader和sampler都有两个

sampler1 = torch.utils.data.sampler.SubsetRandomSampler(
    np.random.permutation(range(len(train_dataset))))
sampler2 = torch.utils.data.sampler.SubsetRandomSampler(
    np.random.permutation(range(len(train_dataset))))

# 训练数据的两个加载器
train_loader1 = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = False,
                                           sampler = sampler1
                                           )
train_loader2 = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = False,
                                           sampler = sampler2
                                           )

# 校验数据和测试数据都各自有两套
val_size = 5000
val_indices1 = range(val_size)
val_indices2 = np.random.permutation(range(val_size))
test_indices1 = range(val_size, len(test_dataset))
test_indices2 = np.random.permutation(test_indices1)
val_sampler1 = torch.utils.data.sampler.SubsetRandomSampler(val_indices1)
val_sampler2 = torch.utils.data.sampler.SubsetRandomSampler(val_indices2)

test_sampler1 = torch.utils.data.sampler.SubsetRandomSampler(test_indices1)
test_sampler2 = torch.utils.data.sampler.SubsetRandomSampler(test_indices2)

val_loader1 = torch.utils.data.DataLoader(dataset = test_dataset,
                                        batch_size = batch_size,
                                        shuffle = False,
                                        sampler = val_sampler1
                                        )
val_loader2 = torch.utils.data.DataLoader(dataset = test_dataset,
                                        batch_size = batch_size,
                                        shuffle = False,
                                        sampler = val_sampler2
                                        )
test_loader1 = torch.utils.data.DataLoader(dataset = test_dataset,
                                         batch_size = batch_size,
                                         shuffle = False,
                                         sampler = test_sampler1
                                         )
test_loader2 = torch.utils.data.DataLoader(dataset = test_dataset,
                                         batch_size = batch_size,
                                         shuffle = False,
                                         sampler = test_sampler2
                                         )


# # MINST Adder

# 为了实现加法器，需要同时处理两个手写体数字图像，并对它进行相应的图像处理
# 因此，网络的架构为两个卷积神经网络，串联上两个全链接层

# In[13]:


depth = [20, 40, 100]
class MINSTAdder(nn.Module):
    def __init__(self):
        super(MINSTAdder, self).__init__()
        #处理第一个图像处理用的卷积网络部件
        self.net_pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm1d(50)


        self.net1_conv1 = nn.Conv2d(1, depth[0], 5, padding = 2)
        self.net1_conv2 = nn.Conv2d(depth[0], depth[1], 5, padding = 2)
        self.net1_conv3 = nn.Conv2d(depth[1], depth[2], 5, padding = 2)
        # self.net1_conv4 = nn.Conv2d(depth[2], depth[3], 5, padding = 2)
        # self.net1_conv5 = nn.Conv2d(depth[3], depth[4], 5, padding = 2)

        #处理第二个图像处理用的卷积网络部件
        self.net2_conv1 = nn.Conv2d(1, depth[0], 5, padding = 2)
        self.net2_conv2 = nn.Conv2d(depth[0], depth[1], 5, padding = 2)
        self.net2_conv3 = nn.Conv2d(depth[1], depth[2], 5, padding = 2)
        # self.net2_conv4 = nn.Conv2d(depth[2], depth[3], 5, padding = 2)
        # self.net2_conv5 = nn.Conv2d(depth[3], depth[4], 5, padding = 2)

        #后面的全连阶层
        self.fc1 = nn.Linear(2 * (image_size // 4 // 2) * (image_size // 4 // 2) * depth[2] , 1000)
        self.fc2 = nn.Linear(1000, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x, y, training = True):
        #第一张图像的处理流程
        x = F.leaky_relu(self.net1_conv1(x))
        x = self.net_pool(x)
        x = F.leaky_relu(self.net1_conv2(x))
        x = self.net_pool(x)
        x = F.leaky_relu(self.net1_conv3(x))
        x = self.net_pool(x)
        x = x.view(-1, (image_size // 4 // 2) * (image_size // 4 // 2)  * depth[2])

        #第二张图像的处理流程
        y = F.leaky_relu(self.net2_conv1(y))
        y = self.net_pool(y)
        y = F.leaky_relu(self.net2_conv2(y))
        y = self.net_pool(y)
        y = F.leaky_relu(self.net2_conv3(y))
        y = self.net_pool(y)
        y = y.view(-1, (image_size // 4 // 2) * (image_size // 4 // 2) * depth[2])

        #将前两部处理得到的张量并列到一起，喂给两层全链接前馈网络，最后输出预测数值
        z = torch.cat((x, y), 1)
        z = self.fc1(z)
        z = F.leaky_relu(z)
        z = F.dropout(z, training=self.training) #以默认为0.5的概率对这一层进行dropout操作
        z = self.fc2(z)
        z = F.leaky_relu(z)
        z = self.bn(z)
#        z = F.dropout(z, training=self.training) #以默认为0.5的概率对这一层进行dropout操作
        z = self.fc3(z)

        return z

# 计算准确度的函数（有多少数字给出了严格的正确输出结果）
def rightness_(y, target):
    out = torch.round(y).type(torch.LongTensor)
    # print('target is ', target)
    # print('out is ',out.data.view_as(target))
    # out = out.data
    # print(type(out))
    # print(type(target))
    # out = out.eq(target.view_as(out)).sum()
    target = target.eq(out.data.view_as(target)).sum()
    out1 = y.size()[0]
    # out = out.data[0]
    # out1 = len(target)
    return(target, out1)


def rightness(y, target):
    out = torch.round(y.view(-1)).type(torch.LongTensor)
    out = out.eq(target).sum()
    out1 = y.size()[0]
    return (out, out1)

# 将网络定义为一个预测器，来对加法的结果进行预测，因此用MSE平均平房误差作为我们的损失函数
net = MINSTAdder()
net = net.cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.5)
results = {}


# In[15]:


# 开始训练循环，本部分代码需要补齐
records = []
for epoch in range(num_epochs):
    train_rights = []
    losses = []
    # 一个关键技术难点是对两个数据加载器进行读取成对儿的数据。我们的办法是通过zip命令，将loader1和2并列在一起，一对一对的读取数据
    for batch_idx, data in enumerate(zip(train_loader1, train_loader2)):
        ((x1, y1), (x2, y2)) = data
        labels = y1 + y2
        # labels = Variable(labels.type(torch.FloatTensor))
        net.train()
        outputs = net(Variable(x1).cuda(), Variable(x2).cuda())
        # loss = criterion(outputs, Variable(labels.type(torch.FloatTensor)).cuda())
        loss = criterion(outputs, Variable(labels.type(torch.FloatTensor)).cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        right = rightness(outputs.data, labels)
        train_rights.append(right)
        losses.append(loss.data.cpu())

        if batch_idx % 100 == 0:
            net.eval()
            val_rights = []
            for idx, data in enumerate(zip(val_loader1, val_loader2)):
                ((x1, y1), (x2, y2)) = data
                outputs = net(Variable(x1).cuda(), Variable(x2).cuda())
                labels = y1+y2
                # labels = Variable(labels.type(torch.FloatTensor))
                right = rightness(outputs.data, labels)
                val_rights.append(right)

            train_r = np.sum([tup[0] for tup in train_rights]) / float(np.sum([tup[1] for tup in train_rights])) * 100.
            val_r = np.sum([tup[0] for tup in val_rights]) / float(np.sum([tup[1] for tup in val_rights])) * 100.

            print('epoch: {} [{}/{} ({:.0f}%)]\t acc on train: {:.4f}%\t acc on val: {:.4f}%\t{}'.format(
                epoch, batch_idx * batch_size, len(train_loader1.dataset), 100. * batch_idx / len(train_loader1),
                train_r, val_r, arrow.utcnow()
                ))
            records.append([np.mean(losses), train_r, val_r])

            # print(epoch, idx, len(train_loader1), np.mean(losses), type(right_ratio))

            # print('第{}周期，第({}/{})个撮，训练loss：{:.2f}, 准确率：{:.2f}'.format(
                            # epoch, idx, len(train_loader1),
                            # np.mean(losses), right_ratio))

            # print(type(train_r[0]), type(train_r[1]))
#            #val_r为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数
#            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
#            print(train_r[0], train_r[1])
#
#            #打印准确率等数值，其中正确率为本训练周期Epoch开始后到目前撮的正确率的平均值
#            print('训练周期: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t训练正确率: {:.2f}%\t校验正确率: {:.2f}%\t{}'.format(
#                epoch, batch_idx * len(data), len(train_loader1.dataset),
#                100. * batch_idx / len(train_loader1), loss.data[0],
#                100. * train_r[0] / train_r[1],
#                100. * val_r[0] / val_r[1],
#                arrow.utcnow()))


# In[ ]:


# 在测试集上运行我们的加法机网络，并测试预测准确度



