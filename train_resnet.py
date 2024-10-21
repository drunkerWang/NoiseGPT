import torch.optim as optim
import torch
import time
import os
import json
import argparse
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from model.resnet import resnet18
from model.preresnet import ResNet18 as preresnet18
from torch.autograd import Variable
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet18", choices=['resnet18', 'preresnet18'])
parser.add_argument("--data_location", type=str, default="/home/wanghaoyu/data/dataset")
parser.add_argument("--dataset", type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument("--train_val_ratio", type=float, default=0.8)
parser.add_argument("--noise_type", type=str, default=None, choices=['symmetric', 'asymmetric'])
parser.add_argument("--noise_level", type=float, default=None)
parser.add_argument("--is_corrected", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--lr", type=float, default=2e-2)
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--milestone", type=int, default=30)
parser.add_argument("--optimizer", type=str, default="SGD", choices=['SGD', 'adam'])
parser.add_argument("--save_ckpt", type=str, default='/home/wanghaoyu/data/result/NoiseGPT/resnet')
parser.add_argument("--cudaid", type=str, default='cuda:0')
parser.add_argument("--in_channel", type=int, default=3)
args = parser.parse_args()
text_template_mapping = {
    'cifar10': 'cifar_template',
    'cifar100': 'cifar_template',
    'imagenet': 'openai_imagenet_template',
}

# 设置超参数
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
DEVICE = torch.device(args.cudaid if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'

# 数据预处理
transformer = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 读取数据
dataset_class = getattr(datasets, args.dataset.upper())
dataset_train = dataset_class(args.data_location, train=True, transform=transformer, target_transform=None, download=True)
if args.is_corrected is not None:
    print(args.is_corrected)
    noise_file = f'/home/wanghaoyu/data/noisy_dataset/CIFAR-10_{args.noise_type}{str(args.noise_level)}_iter0-50000_corrected.pt'
    noisy_targets = torch.load(f'/home/wanghaoyu/data/noisy_dataset/CIFAR-10_{args.noise_type}{str(args.noise_level)}_iter0-50000_corrected.pt')
    dataset_train.targets = noisy_targets
else:
    print('noise_type')
    noise_file = f'/home/wanghaoyu/data/noisy_dataset/CIFAR-10_{args.noise_type}{str(args.noise_level)}.pt'
    noisy_targets = torch.load(f'/home/wanghaoyu/data/noisy_dataset/CIFAR-10_{args.noise_type}{str(args.noise_level)}.pt')
    dataset_train.targets = noisy_targets
    pass
dataset_test = dataset_class(args.data_location, train=False, transform=transformer, target_transform=None, download=True)
 
# 导入数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
modellr = args.lr
 
# 实例化模型并且移动到GPU
if args.dataset == 'cifar10':
    num_class = 10
elif args.dataset == 'cifar100':
    um_class = 100
if args.model == 'resnet18':
    model = resnet18(pretrained=False, in_channel=args.in_channel, num_classes=num_class) 
elif args.model == 'preresnet18': 
    model = preresnet18(num_classes=num_class)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()  # since the output of network is by log softmax

if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=modellr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=modellr, weight_decay=args.weight_dacay)
milestone = [int(args.milestone*(i+1)) for i in range(args.epochs//args.milestone-1)]
# exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=args.gamma)
cos_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs+10, eta_min=0, last_epoch=-1, verbose=False)

 
# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    train_correct = 0
    train_total = len(train_loader.dataset)
    train_loss = 0
    print("current train data size:", len(train_loader.dataset))

    model.train()

    for batch_idx, (image, target) in enumerate(train_loader):

        image, target = Variable(image).to(device), Variable(target).to(device)

        output = model(image)
        # log_output = torch.log_softmax(output, 1)
        # loss = criterion(log_output, target)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        train_correct += predicted.eq(target).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tlr: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(image), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), cos_lr_scheduler.get_last_lr()[0]))
    
    train_acc = train_correct / train_total * 100.
    print(f'epoch: {epoch}\ttrain accuracy: {train_acc:.2f}\ttrain loss: {train_loss:.4f}') 
 
def val(model, device, test_loader):
    model.eval()

    test_loss = 0
    test_correct = 0
    test_total = len(test_loader.dataset)

    with torch.no_grad():
        for image, target in test_loader:
            image, target = Variable(image).to(device), Variable(target).to(device)

            output = model(image)
            # log_output = torch.log_softmax(output, 1)
            # loss = criterion(log_output, target)
            loss = criterion(output, target)

            test_loss += loss.item()

            _, predicted = output.max(1)
            # print(predicted)
            test_correct += predicted.eq(target).sum().item()
            # print(predicted.eq(target).sum().item())
            # print(target)

        test_acc = test_correct / test_total * 100.
        avg_loss = test_loss / len(test_loader)

        print('[val set] Average Loss: {:.4f}\tAccuracy: {:.2f} ({:.0f}/{:.0f})\n'.format(avg_loss, test_acc, test_correct, test_total)) 
        return test_acc, avg_loss
 
 
# 训练
test_acc_history = []
test_loss_history = []
for epoch in range(1, EPOCHS + 1):
    print(noise_file)
    train(model, DEVICE, train_loader, optimizer, epoch)
    # exp_lr_scheduler.step()
    cos_lr_scheduler.step()

    test_acc, test_loss = val(model, DEVICE, test_loader)
    test_acc_history.append(test_acc)
    test_loss_history.append(test_loss)

timestamp = time.time()
time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))

plt.figure(figsize=(8, 8))
x = range(1, int(args.epochs+1))
plt.plot(x, test_acc_history, color=(139/256,181/256,209/256), lw=2, label='test accuracy curve')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('test accuracy')
plt.legend(loc='lower right')
plt.savefig(f'/home/wanghaoyu/data/result/NoiseGPT/resnet/{args.dataset}_{args.noise_type}{str(args.noise_level)}_corrected{args.is_corrected}_accuracy_{time_str}.png'
            , format='png', dpi=500)

plt.figure(figsize=(8, 8))
x = range(1, int(args.epochs+1))
plt.plot(x, test_loss_history, color=(139/256,181/256,209/256), lw=2, label='test loss curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('test loss')
plt.legend(loc='lower right')
plt.savefig(f'/home/wanghaoyu/data/result/NoiseGPT/resnet/{args.dataset}_{args.noise_type}{str(args.noise_level)}_corrected{args.is_corrected}_loss_{time_str}.png'
            , format='png', dpi=500)

save_path = os.path.join(args.save_ckpt, 
                         f'{args.dataset}_{args.noise_type}{str(args.noise_level)}_corrected{args.is_corrected}__{time_str}.pth'
                         )
torch.save(model, save_path)

output_dict={'test acc': test_acc, 'test loss': test_loss}
with open(f'/home/wanghaoyu/data/result/NoiseGPT/resnet/train_history.json', 'a') as f:
    json.dump(output_dict, f)
    json.dump(vars(args), f)
    f.write('\n')
