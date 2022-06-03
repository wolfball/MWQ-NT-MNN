from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import random
import copy
import numpy as np
import pdb
import os


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def create_qnn(model_accurate, quantization_threshold, quantized_value, std, mean, n_values):
    model_qnn = copy.deepcopy(model_accurate)
    quantize_weights(model_qnn, quantization_threshold=quantization_threshold, quantized_value=quantized_value, n_values=n_values)
    add_noise_to_weights(model_qnn, original=model_accurate, std=std, mean=mean)
    return model_qnn


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model_qnn = create_qnn(model, quantization_threshold=args.quantization_threshold,
                               quantized_value=args.quantized_value, std=args.std, mean=args.mean,
                               n_values=args.n_values)
        output = model_qnn(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        modify_gradient(model, model_qnn, args.grad_threshold)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(args, model, device, test_loader, std):
    model.eval()
    test_loss = 0
    correct = 0

    old_params = [p.detach().cpu().numpy() for p in model.parameters()]
    model_qnn = create_qnn(model, quantization_threshold=args.quantization_threshold,
                           quantized_value=args.quantized_value, std=args.std, mean=args.mean,
                           n_values=args.n_values)
    new_params = [p.detach().cpu().numpy() for p in model_qnn.parameters()]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model_qnn(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset), {'old':old_params, 'new':new_params}


def quantize_weights(model, quantization_threshold, quantized_value, n_values):
    assert quantization_threshold >= 0, quantizedvalue >= 0
    n_ = (n_values - 1) // 2
    with torch.no_grad():
        for param in model.parameters():
            param_copy = param.clone().detach()
            param_copy[param.abs() <= quantization_threshold] = 0
            for i in range(1, n_+1):
                param_copy[param > i * quantization_threshold] = i * quantized_value
                param_copy[param < -i * quantization_threshold] = -i * quantized_value
            param.copy_(param_copy)


def add_noise_to_weights(model, original=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), std=0.5, mean=0):
    """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * 0.1)
    """
    gassian_kernel = torch.distributions.Normal(mean, std)
    with torch.no_grad():
        if original is None:
            for param in model.parameters():
                noise = torch.exp(gassian_kernel.sample(param.size())).to(device)
                param.mul_(noise)
        else:
            for param, param_original in zip(model.parameters(), original.parameters()):
                noise = torch.exp(gassian_kernel.sample(param.size())).to(device)
                param.mul_(noise)
                param_original.grad = noise


def modify_gradient(model_accurate, model_qnn, grad_threshold):
    assert grad_threshold >= 0
    for param_accurate, param_qnn in zip(model_accurate.parameters(), model_qnn.parameters()):
        mask = param_accurate.clone().detach()
        mask[param_accurate.abs() <= grad_threshold] = 1
        mask[param_accurate.abs() > grad_threshold] = 0
        param_accurate.grad.mul_(param_qnn.grad * mask)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
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
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--std', type=float, default=1.0,
                        help='Noise std')
    parser.add_argument('--mean', type=float, default=0,
                        help='Noise mean')
    parser.add_argument('--quantization_threshold', type=float, default=0.01,
                        help='Quantization threshold')
    parser.add_argument('--quantized_value', type=float, default=0.01,
                        help='Quantized value')
    parser.add_argument('--grad_threshold', type=float, default=0.5,
                        help='Gradient clip threshold')
    parser.add_argument('--n_values', type=int, default=7,
                        help='Gradient clip threshold')
    parser.add_argument('--expname', type=str, default='exp')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA id')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device(f"cuda:{args.cuda}" if use_cuda else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    os.makedirs(os.path.join('res', args.expname), exist_ok=True)
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    test_acc = []
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        acc, params = test(args, model, device, test_loader, args.std)
        torch.save(params, os.path.join('res', args.expname, f'params{epoch}.pt'))
        test_acc.append(acc)
        scheduler.step()


    torch.save(test_acc, os.path.join('res', args.expname, 'testacc.pt'))
    print(test_acc)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
