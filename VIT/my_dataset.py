from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        # 获取数据集的图像个数
        return len(self.images_path)

    # self 代表类的实例对象，item 是要获取的索引或键
    def __getitem__(self, item):
        # 根据图像路径，使用__getitem__函数根据索引读入单个图像
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        # 检测是否为RGB图像，如果不是，则需要更改网络结构和图像预处理规范化方式
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        # 根据索引读入单个标签
        label = self.images_class[item]

        # 根据数据集类型对图像进行相应的预处理
        if self.transform is not None:
            img = self.transform(img)
        # 返回预处理后的图像和对应的标签
        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
