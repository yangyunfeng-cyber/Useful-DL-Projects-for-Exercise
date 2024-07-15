import os
import time
import datetime

import torch

from src import UNet
from src import ULite
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import DriveDataset
import transforms as T

# 定义训练集图像的预处理方式
class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]   # 对图像的短边(长和宽中最短的)进行随机缩放以适应不同图像输入尺寸，缩放范围为【min_size, max_size】
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))  # 加入水平翻转
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))  # 加入垂直翻转
        trans.extend([
            T.RandomCrop(crop_size),   # 对图像进行随机裁剪
            T.ToTensor(),  # 将数组矩阵转换为tensor类型，规范化到【0，1】范围
            T.Normalize(mean=mean, std=std),  # 加入图像归一化，并定义均值和标准差，RGB三通道的
        ])
        # trans是一个列表类型，包含各种了变换，将这些变换组成一个compose变换，注意transforms.Compose()函数需要接收一个列表类型
        self.transforms = T.Compose(trans)

    # 使用__call__()函数来调用transforms变换
    def __call__(self, img, target):
        return self.transforms(img, target)  # target是指标签图像，img是指待分割图像


# 定义验证集的图像预处理组合类，比较简单，只有张量化和规范化两个操作，这里规范化使用的是ImageNet推荐的参数，注意这种做法是针对彩色图像
class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


# 定义一个函数根据数据集的类型来调用对应的数据集处理类
def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480
    # 检查train是否为True
    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)

# 定义模型创建函数，实例化UNet类创建模型，传入通道数，分割类别数，
def create_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=64)   # 输入通道数为3，分类类别数为2，
    # model = MobileV3Unet(num_classes=num_classes, pretrain_backbone=True)
    # model = VGG16UNet(num_classes=num_classes, pretrain_backbone=True)
    # model = ULite(in_channels=3, num_classes=num_classes)

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])    # 如果batch_size>1, 线程数num_workers取min(cpu核数,batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, args.num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat, dice = evaluate(model, val_loader, device=device, num_classes=args.num_classes)

        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "save_weights/best_model_UNet.pth")
        else:
            torch.save(save_file, "./save_weights/model_amp_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=300, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')

    # 从上次训练停止的地方重新开始训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # 保存最佳模型
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # 混合精度训练参数，用于加快模型训练
    parser.add_argument("--amp", default=True, type=bool,  help="Use torch.cuda.amp for mixed precision training")

    # 解析命令行参数，并将解析结果保存在args对象中
    args = parser.parse_args()
    # 返回解析结果
    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
