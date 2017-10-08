# coding: utf-8

import os
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # dont show, just save

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import torchvision
from torchvision import utils as vutil

# from IPython.display import display

use_cuda = torch.cuda.is_available()

print("GPU Avaialble: ", use_cuda)

# init path
path = {
    'datasets': os.path.join(os.getcwd(), 'datasets'),
    'models': os.path.join(os.getcwd(), 'models'),
    'images': os.path.join(os.getcwd(), 'images')
}

for k, p in path.items():
    if not os.path.isdir(p):
        os.mkdir(p)

print("==== Path OK ====")
transform_sets = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

print(path)
params = {
    'datasets': {
        'root': path['datasets'],
        'download': True,
        'transform': transform_sets
    }
}

data = {
    'train': torchvision.datasets.MNIST(train=True, **params['datasets']),
    'test': torchvision.datasets.MNIST(train=False, **params['datasets']),
}


# set dataloader

batch_size = 100
n_class = 11
dataloader = {}

# dataloader-train

dataloader['train'] = torch.utils.data.DataLoader(dataset=data['train'], batch_size=batch_size, shuffle=True)

# dataloader-valid & dataloader-test

# auxiliary components

indicies = {
    'valid': range(len(data['test'])//2),
    'test': range(len(data['test'])//2, len(data['test']))
}
sampler = {
    'valid': torch.utils.data.sampler.SubsetRandomSampler(indices=indicies['valid']),
    'test': torch.utils.data.sampler.SubsetRandomSampler(indices=indicies['test'])
}

### dataloder definition

dataloader['valid'] = torch.utils.data.DataLoader(dataset=data['test'],
                                                  batch_size=batch_size,
                                                  sampler=sampler['valid'])

dataloader['test'] = torch.utils.data.DataLoader(dataset=data['test'],
                                                 batch_size=batch_size,
                                                 sampler=sampler['test'])


# draw images dist for train, validation and test dataset separatly
def vis_dist_dataloader(cls):
    count = {i: 0 for i in range(n_class)}
    for i, data in enumerate(dataloader[cls]):
        xs, ys = data
        for y in ys:
            count[y] += 1

    count = [count[i] for i in range(n_class)]
    # display(pd.DataFrame(data=count))
    print(pd.DataFrame(data=count))
    plt.bar(left=np.arange(n_class), height=count)
    plt.xlabel('digit_class')
    plt.ylabel('count_total_samples')
    plt.title('simple stat: data[{}]'.format(cls))
    filename_base = 'stat-{}'.format(cls)
    filename_full = os.path.join(path['images'], filename_base)
    plt.savefig(filename_full)


vis_dist_dataloader('train')
vis_dist_dataloader('valid')
vis_dist_dataloader('test')


input_dim = n_class
num_features = 64
num_channels = 1


class ModelG(nn.Module):

    def __init__(self):
        super(ModelG, self).__init__()
        self.deconv00 = nn.ConvTranspose2d(in_channels=input_dim,
                                           out_channels=num_features*2,
                                           kernel_size=5,
                                           stride=2, padding=0) #, bias=False)
        self.deconv01 = nn.ConvTranspose2d(in_channels=num_features*2, out_channels=num_features,
                                           kernel_size=5, stride=2, padding=0) #, bias=False)
        self.deconv02 = nn.ConvTranspose2d(in_channels=num_features, out_channels=num_channels,
                                           kernel_size=4, stride=2, padding=0) #, bias=False)
        self.bn00 = nn.BatchNorm2d(num_features=num_features*2)
        self.bn01 = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x, training=True):
        x = self.bn00(self.deconv00(x))
#         print('deconv0: ', x.data.size())
        x = F.leaky_relu(x, negative_slope=.2)
        x = self.bn01(self.deconv01(x))
#         print('deconv1: ', x.data.size())
        x = F.leaky_relu(x, negative_slope=.2)
        x = self.deconv02(x)
#         print('deconv2: ', x.data.size())
        output = F.tanh(x)

        return output


class ModelD(nn.Module):

    def __init__(self):
        super(ModelD, self).__init__()
        self.conv00 = nn.Conv2d(in_channels=num_channels, out_channels=num_features,
                                kernel_size=5, stride=2, padding=0) #, bias=False)
        self.bn00 = nn.BatchNorm2d(num_features=num_features)
        self.conv01 = nn.Conv2d(in_channels=num_features, out_channels=num_features*2,
                                kernel_size=5, stride=2, padding=0) #, bias=False)
        self.bn01 = nn.BatchNorm2d(num_features=num_features*2)
        self.fc00 = nn.Linear(in_features=(num_features*2 * 4 * 4), out_features=num_features)
        self.fc01 = nn.Linear(in_features=num_features, out_features=n_class)

    def forward(self, x, training=True):
        x = self.bn00(F.leaky_relu(self.conv00(x), negative_slope=.2))
        x = self.bn01(F.leaky_relu(self.conv01(x), negative_slope=.2))
        x = x.view(-1, num_features*2 * 4 * 4)
        x = F.leaky_relu(self.fc00(x), negative_slope=.2)
        x = self.fc01(x)
        output = F.log_softmax(x)

        return output



def weight_init(m):
    if isinstance(m, ModelG) or isinstance(m, ModelD):
        return
    try:
        class_name = m.__class__.name
    except:
        class_name = m.__str__()
    if class_name.find('conv') != -1:
        m.weight.data.normal_(0, 0.02)
    if class_name.find('norm') != -1:
        m.weight.data.normal_(1.0, 0.02)


image_size = 28

def make_show(img):
    img = img.data.expand(batch_size, 3, image_size, image_size)
    return img



def imshow(inp, title=None):
    """Imshow for Tensor."""
    if inp.size()[0] > 1:
        inp = inp.numpy().transpose((1, 2, 0))
    else:
        inp = inp[0].numpy()
    mvalue = np.amin(inp)
    maxvalue = np.amax(inp)
    if maxvalue > mvalue:
        inp = (inp - mvalue)/(maxvalue - mvalue)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated



#  gan 11 classes


def gen_noise(ys, batch_size=batch_size, input_dim=input_dim):
    noise = np.random.randn(batch_size, input_dim, 1, 1)

    new_noise = []
    for i_b, y in enumerate(ys):
        base = noise[i_b].reshape(-1)
        item = np.zeros(input_dim)
        whichone = y.cpu().data.numpy()[0] if use_cuda else y.data.numpy()[0]
        item[whichone] = 10
        full = (base + item).reshape(input_dim, 1, 1)
        new_noise.append([full])

    noise = np.concatenate(new_noise).astype(np.float32)

    return torch.from_numpy(noise)


# init: netG(Generator)

netG = ModelG().cuda() if use_cuda else ModelG()
netG.apply(weight_init)
optG = torch.optim.Adam(netG.parameters(), lr=0.000002,betas=(0.5,0.999))#, lr=0.0002,betas=(0.5,0.999)) # DIFFERENT from example code

# init: netD(Discriminator)

netD = ModelD().cuda() if use_cuda else ModelD()
netD.apply(weight_init)
optD = torch.optim.SGD(netD.parameters(), lr=0.0002)#, lr=0.0002,betas=(0.5,0.999))

# inputs...

# noise = Variable(torch.FloatTensor(batch_size, 1, 1, 1))
# fixed_noise = torch.from_numpy(np.random.choice(n_class-1, batch_size))
# fixed_noise = fixed_noise.view(batch_size, 1, 1, 1)

# noise, fixed_noise = (noise.cuda(), fixed_noise.cuda()) if use_cuda else (noise, fixed_noise)


# loss...

criterion = nn.CrossEntropyLoss()
error_G = None
num_epochs = 100
results = []

threshold = 0.5 # DIFFERENT from example code

for epoch in range(num_epochs):

    for batch_idx, (xs_raw, ys_raw) in enumerate(dataloader['train']):

        # 1. REAL pic
        optD.zero_grad()

        ## use cuda?
        label_shape = ys_raw.size()
        label_type = torch.cuda.LongTensor if use_cuda else torch.LongTensor
        xs, ys = (xs_raw.cuda(), ys_raw.cuda()) if use_cuda else (xs_raw, ys_raw)

        ## pack in Variable
        xs, ys = Variable(xs), Variable(ys)

        netD.train() # set training state
        output = netD(xs)
        error_real = criterion(output, ys)
        error_real.backward() # BP 01 -> netD

        # 2. Generate pic
#         noise = torch.FloatTensor(batch_size, input_dim, 1, 1)
#         noise = torch.normal(torch.zeros(label_shape), torch.ones(label_shape))
#         noise = (noise + ys.type(torch.FloatTensor).data) / 10.0
#         noise = Variable(noise)
#         noise = noise.unsqueeze(1).expand(noise.size()[0], input_dim)
#         noise.data.resize_(noise.data.size()[0], input_dim, 1, 1)
        noise = Variable(gen_noise(ys))
        noise.data = noise.data.cuda() if use_cuda else noise.data #gpu transform
        fake_pic = netG(noise).detach()
        output2 = netD(fake_pic)
#         label = Variable(torch.zeros(label_shape).type(label_type))
        label = Variable(torch.from_numpy(np.array([n_class-1] * label_shape[0])).type(label_type))
        error_fake = criterion(output2, label)
        error_fake.backward()
        error_D = error_real + error_fake
        optD.step() # start to optimize: Discriminator

        # train Generator alone:
#         error_G = Variable(torch.zeros(1))
        if (error_G is None) or (np.random.rand() < threshold):
            optG.zero_grad()
            netG.train()
#             noise = torch.normal(torch.zeros(label_shape), torch.ones(label_shape))
#             noise = (noise + ys.type(torch.FloatTensor).data) / 10.0
#             noise = Variable(noise)
#             noise = noise.unsqueeze(1).expand(noise.size()[0], input_dim)
#             noise.data.resize_(noise.data.size()[0], input_dim, 1, 1)
            noise = Variable(gen_noise(ys))
            noise.data = noise.data.cuda() if use_cuda else noise.data #gpu transform
            fake_pic = netG(noise)
            output3 = netD(fake_pic)
#             if use_cuda:
#                 m_v,m_i = torch.max(output3.cpu(), 1)
#             else:
#                 m_v,m_i = torch.max(output3, 1)
#             if m_i.numpy() < (n_class-1):
#                 print('good')
            error_G = criterion(output3, ys.type(label_type))
            error_G.backward()
            optG.step() # start to optimize: Generator

        error_D, error_G = (error_D.cpu(), error_G.cpu()) if use_cuda else (error_D, error_G)
        results.append([float(error_D.data.numpy()), float(error_G.data.numpy())])

        if batch_idx % 100 == 0:
#             print('output:', output.cpu().data if use_cuda else output.data)
#             print('output2:', output2.cpu().data if use_cuda else output2.data)
#             print('error_real: {} \t error_fake: {}'.format(error_real, error_fake))
            print ('epoch {}，batch at {}/{}, Classifier Loss:{:.2f}, Generator Loss:{:.2f}'.format(
                epoch,batch_idx,len(dataloader['train']), error_D.data[0], error_G.data[0]))

    # Generate some fake pics and evaluate
    netG.eval()
    fixed_noise = torch.from_numpy(np.random.choice(n_class-1, batch_size))
    fixed_noise = fixed_noise.type(torch.FloatTensor)
    fixed_noise = fixed_noise.unsqueeze(1).expand(fixed_noise.size()[0], input_dim)
    fixed_noise = Variable(fixed_noise)
    fixed_noise.data.resize_(fixed_noise.data.size()[0], input_dim, 1, 1)
    fixed_noise.data = fixed_noise.data.cuda() if use_cuda else fixed_noise.data #gpu transform

    fake_u = netG(fixed_noise)
    fake_u = fake_u.cpu() if use_cuda else fake_u
    img = make_show(fake_u)

    # pick some pics and save
    vutil.save_image(img, os.path.join(path['images'], "Generator_pic_epoch%s.png" % (epoch)))

    # save net state
    if epoch % 10 == 0:
        torch.save(netG.state_dict(), os.path.join(path['models'], 'netG_epoch_%d' % (epoch)))
        torch.save(netD.state_dict(), os.path.join(path['models'], 'netD_epoch_%d' % (epoch)))



# 预测曲线
plt.figure(figsize = (10, 7))
plt.plot([i[1] for i in results], '.', label = 'Generator', alpha = 0.5)
plt.plot([i[0] for i in results], '.', label = 'Discreminator', alpha = 0.5)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()


fixed_noise = Variable(gen_noise(ys))
fixed_noise.data = fixed_noise.data.cuda() if use_cuda else noise.data #gpu transform
fixed_labels = torch.max(fixed_noise, 1)[1]
fixed_labels = fixed_labels.resize(batch_size)

fake_u = netG(fixed_noise)
fake_u = fake_u.cpu() if use_cuda else fake_u
img = fake_u #.expand(sample_size, 3, image_size, image_size) #将张量转化成可绘制的图像
#print(img.size())
fig = plt.figure(figsize = (15, 15))

sample_size = batch_size
for i in range(sample_size):
    ax = plt.subplot(10, 10, i+1 if i+1 <= 100 else 100)
    ax.axis('off')
    thislabel = fixed_labels[i].cpu().data.numpy()[0] if use_cuda else fixed_labels[i].data.numpy()[0]
    imshow(img[i].data, thislabel)
