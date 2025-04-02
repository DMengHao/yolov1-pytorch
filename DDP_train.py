import torch.distributed as dist
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from yoloData import yoloDataset
from yoloLoss import yoloLoss
from new_resnet import resnet50
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os

file_root = 'VOCdevkit/VOC2007/JPEGImages/'
batch_size = 32
learning_rate = 0.001
num_epochs = 200

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '10.145.106.37'
    os.environ['MASTER_PORT'] = '8098'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.manual_seed(42+rank) # 不同进程设置不同的随机种子
    torch.cuda.manual_seed(42+rank)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train(rank, world_size):
    setup(rank=rank, world_size=world_size)
    # 数据加载
    train_dataset = yoloDataset(img_root=file_root, list_file='voctrain.txt', train=True, transform=[transforms.ToTensor()])
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              sampler=sampler,
                              num_workers=0, # 根据CPU核心数进行调整
                              pin_memory=True, # 加速数据到GPU的传输
                              drop_last=True) # 避免最后批次尺寸不匹配
    # 模型初始化
    model = resnet50()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    # 损失函数和优化器
    criterion = yoloLoss(7, 2, 5, 0.5)
    criterion = criterion.to(rank)

    optimizer = torch.optim.SGD(  # 定义优化器  “随机梯度下降”
        model.parameters(),  # net.parameters() 为什么不用这个???
        lr=learning_rate,
        momentum=0.9,  # 即更新的时候在一定程度上保留之前更新的方向  可以在一定程度上增加稳定性，从而学习地更快
        weight_decay=5e-4)  # L2正则化理论中出现的概念
    # 学习率调度器
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1**(epoch//30))

    for epoch in range(num_epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0.0

        # 学习率更新（所有进程同步）
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if rank==0:
            print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
            print('Learning Rate for this epoch: {}'.format(current_lr))

        for i, (images, target) in enumerate(train_loader):
            images, target = images.to(rank, non_blocking=True), target.to(rank, non_blocking=True)
            optimizer.zero_grad(set_to_none=True) # 更高效梯度清零
            pred = model(images)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if rank == 0 and i%5==0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (epoch + 1, num_epochs,
                                                                                      i + 1, len(train_loader),
                                                                                      loss.item(),
                                                                                      total_loss / (i + 1)))

        torch.save(model.state_dict(), './weights/YOLOv1_VOC_DDP.pth')

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
