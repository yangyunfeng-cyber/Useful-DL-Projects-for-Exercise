import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class DriveDataset(Dataset):  # 继承Dataset类
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "training" if train else "test"  # 根据train这个布尔类型确定需要处理的是训练集还是测试集
        data_root = os.path.join(root, "DRIVE", self.flag)  # 得到数据集根目录
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."   # 判断路径是否存在
        self.transforms = transforms   # 初始化图像变换操作
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]   # 遍历图像文件夹获取每个图像的文件名
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]  # 获取图像路径
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")  # 获取手动标签的路径
                       for i in img_names]
        # 检查手动标签文件是否存在
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
        # 获取分割的ROI区域掩码
        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names]
        # check files
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')  # 加载图像，并转换为RGB模式
        manual = Image.open(self.manual[idx]).convert('L')   # 加载手动标注图像，并转换为灰度模式
        manual = np.array(manual) / 255   # 进行归一化操作
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')  # 加载ROI图像，并转换为灰度模式
        roi_mask = 255 - np.array(roi_mask)   # 对图像数组取反，使用这个方法将背景和前景颜色反转，白色是255，黑色是0，反转后ROI变成了内黑外白
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)   # 将手动标注图像和反转后的ROI图像相加，使用np.clip()将像素值控制在0-255范围，

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    # 获取图像数据集长度
    def __len__(self):
        return len(self.img_list)

    # 用于将批量的图像和标签数据合并为一个批张量。
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))  # 将批量数据拆分为图像和标签两个列表
        batched_imgs = cat_list(images, fill_value=0)  # 使用 cat_list() 函数将图像和标签列表合并成张量。用于将列表中的 PIL 图像数据堆叠成张量，
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))  # 找到图像中最大的形状，以元组形式返回给max_size
    batch_shape = (len(images),) + max_size   # 计算出堆叠后的张量形状，包括批量大小和图像大小两个维度
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)  # 创建一个新的空白张量 batched_imgs，其形状与 batch_shape 相同，并将其填充为指定的填充值 fill_value
    for img, pad_img in zip(images, batched_imgs):  # 使用 zip() 函数将输入列表中的每个图像与其对应的空白张量进行拼接，以得到一个完整的张量。
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)  # 将每个图像按照其实际大小插入到空白张量的左上角，以保持图像的相对位置不变。
    return batched_imgs

