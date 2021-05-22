from __future__ import print_function
import argparse
import torch
import spconv
from spconv.modules import SparseModule
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

class PrintLayer(nn.Module):
    def __init__(self, id):
        super(PrintLayer, self).__init__()
        self.id = id
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(self.id, type(x), x.shape)
        return x

class SparseZeroPad2d(SparseModule):
    def __init__(self, pad):
        super(SparseModule, self).__init__()
        self.pad = pad

    def forward(self, x):
        w, h = x.spatial_shape
        x.spatial_shape = torch.Size([w + 2 * self.pad, h + 2 * self.pad])
        x.indices[:, 1] += self.pad
        x.indices[:, 2] += self.pad
        return x

class SparseScale2d(SparseModule):
    def __init__(self, scale):
        super(SparseModule, self).__init__()
        self.scale = scale

    def forward(self, x):
        w, h = x.spatial_shape
        x.spatial_shape = torch.Size([w * self.scale, h * self.scale])
        x.indices[:, 1:] = x.indices[:, 1:] * self.scale
        return x

def aug_data(data: torch.Tensor):
    layers = [data, torch.zeros_like(data), torch.zeros_like(data)]
    random.shuffle(layers)
    return torch.cat(layers, 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.szp = SparseZeroPad2d(1)
        self.bn = nn.BatchNorm1d(3)
        self.cv1 = spconv.SparseConv2d(3, 3, 5, 5)
        self.ss1 = SparseScale2d(5)
        self.cv2 = spconv.SparseConv2d(3, 3, 5, 5)
        self.ss2 = SparseScale2d(5)
        self.mp = spconv.SparseMaxPool2d(2, 2)
        self.td = spconv.ToDense()

        self.net = spconv.SparseSequential(
            SparseZeroPad2d(1),
            nn.BatchNorm1d(3),
            spconv.SparseConv2d(3, 32, 3, 1),
            nn.ReLU(),
            spconv.SparseConv2d(32, 64, 3, 1),
            nn.ReLU(),
            spconv.SparseMaxPool2d(2, 2),
            spconv.ToDense(), 
        )

        self.fc1 = nn.Linear(10816, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)


    def forward(self, x: torch.Tensor):
        # x: [N, 28, 28, 1], must be NHWC tensor
        x = x.permute(0, 2, 3, 1)
        # x = x.reshape(-1, 28, 28, 1)
        # print("Before shape:",x.shape)
        # create SparseConvTensor manually: see SparseConvTensor.from_dense
        x_sp = spconv.SparseConvTensor.from_dense(x)        

        plt.title("Before pad " + str(x_sp.dense().shape))
        plt.imshow(x_sp.dense().permute(0, 2, 3, 1).cpu().detach().numpy()[0])
        plt.show()
        
        x_sp = self.szp(x_sp)
        plt.title("After pad"+ str(x_sp.dense().shape))
        plt.imshow(x_sp.dense().permute(0, 2, 3, 1).cpu().detach().numpy()[0])
        plt.show()
        x_sp.features = self.bn(x_sp.features)
        print("After norm"+ str(x_sp.dense().shape))
        plt.title("After norm"+ str(x_sp.dense().shape))
        plt.imshow(x_sp.dense().permute(0, 2, 3, 1).cpu().detach().numpy()[0])
        plt.show()
        x_sp = self.cv1(x_sp)
        x_sp = self.ss1(x_sp)
        print("After cv1"+ str(x_sp.dense().shape))
        plt.title("After cv1"+ str(x_sp.dense().shape))
        plt.imshow(x_sp.dense().permute(0, 2, 3, 1).cpu().detach().numpy()[0])
        plt.show()
        x_sp.features = F.relu(x_sp.features)
        plt.title("After relu"+ str(x_sp.dense().shape))
        plt.imshow(x_sp.dense().permute(0, 2, 3, 1).cpu().detach().numpy()[0])
        plt.show()
        x_sp = self.cv2(x_sp)
        x_sp = self.ss2(x_sp)
        plt.title("After cv2"+ str(x_sp.dense().shape))
        plt.imshow(x_sp.dense().permute(0, 2, 3, 1).cpu().detach().numpy()[0])
        plt.show()
        x_sp.features = F.relu(x_sp.features)
        plt.title("After relu"+ str(x_sp.dense().shape))
        plt.imshow(x_sp.dense().permute(0, 2, 3, 1).cpu().detach().numpy()[0])
        plt.show()
        x_sp = self.mp(x_sp)
        plt.title("After max_pool"+ str(x_sp.dense().shape))
        plt.imshow(x_sp.dense().permute(0, 2, 3, 1).cpu().detach().numpy()[0])
        plt.show()
        x = self.td(x_sp)
        plt.title("After x"+ str(x.shape))
        plt.imshow(x_sp.dense().permute(0, 2, 3, 1).cpu().detach().numpy()[0])
        plt.show()

        # x = self.net(x_sp)

        x = torch.flatten(x, 1)
        print(x.shape)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    print("device:", device)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print((data != 0).sum())
        # print(data.shape)
        # im_data = data.permute(0,2,3,1)
        # plt.imshow(im_data.detach().numpy().reshape(*im_data.shape[1:]))
        # plt.show()
        data = aug_data(data)
        # im_data = data.permute(0,2,3,1)
        # plt.imshow(im_data.detach().numpy()[0])
        # plt.show()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = aug_data(data)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # here we remove norm to get sparse tensor with lots of zeros
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           # here we remove norm to get sparse tensor with lots of zeros
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
