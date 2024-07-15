import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


from my_dataset import MyDataSet
from models.vit_model import vit_base_patch16_224_in21k as create_model

from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    # 检测cuda是否可用并定义device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 如果没有weights文件夹，则创建
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 创建一个用于写入tensorboard日志的对象tb_writer,用来记录各种指标，可用使用 tb_writer.add_scalar() 方法来添加标量数据，
    # 使用 tb_writer.add_image() 来添加图像数据，使用 tb_writer.add_graph() 来添加模型结构图，
    # 使用 tb_writer.close() 来关闭 SummaryWriter 对象
    tb_writer = SummaryWriter()
    # 获得训练集和验证集的图像和标签的绝对路径，列表类型的数据
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 定于数据集预处理的方式
    data_transform = {
        "train": transforms.Compose(
            # 图像随机中心裁剪为224*224
            [transforms.RandomResizedCrop(224),
             # 随机翻转
             transforms.RandomHorizontalFlip(),
             # 张量化
             transforms.ToTensor(),
             # 规范化
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        ),
        "val": transforms.Compose(
            # 图像大小调整为256*256
            [transforms.Resize(256),
             # 中心裁剪为224*224
             transforms.CenterCrop(224),
             # 张量化
             transforms.ToTensor(),
             # 规范化
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    # 获取BatchSize的大小
    batch_size = args.batch_size
    # 确定使用进程的个数，min函数中包含三个元素，[cpu的核心数目；如果BatchSize大于1，则取BatchSize，否则取0；8]
    # 根据实际情况，我的电脑中cpu内核为24，此处BatchSize设置为8。则min[24,8,8]=8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # 使用 PyTorch 中的 DataLoader 类来创建一个数据加载器对象 train_loader
    train_loader = torch.utils.data.DataLoader(
                                               # 提供训练集数据
                                               train_dataset,
                                               # 设置BatchSize大小
                                               batch_size=batch_size,
                                               # 对数据进行随机重排序，保证模型不受到数据排序方式的影响
                                               shuffle=True,
                                               # 将数据加载到cuda固定内存中，加快数据传输速度；但同时也会消耗更多的显存
                                               pin_memory=True,
                                               # 设置加载数据的线程数，线程越多加载数据越快，但同时会消耗更多的内存资源
                                               num_workers=nw,
                                               # collate_fn 是用于处理从数据集中取出的每个样本，并将它们组装成一个批量数据的函数。
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 实例化VIT模型, 确定分类的数量，
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)


    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./dataset/flower_photos")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)  # 设置为True则只训练最后一层全连接层，训练速度会加快很多
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
