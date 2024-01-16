# 经典项目代码及原理
本项目提供可以复现的经典网络项目，并提供详细的代码注释，若无法复现请考虑是否与项目第三方库版本有出入。

1.VIT图像分类模型--Vision Transformer文件夹
  项目环境配置： pytorch和cuda版本： 2.0.0+cu118； torchvision版本:0.15.1；其它详见文件夹中的requirements.txt
  论文及详细代码解读：见微信公众号- “Medical AI 研修库”，待更新....
  论文亮点：文章于2020年发表，将NLP中的transformer结构引入到了视觉领域，VIT模型证明了仅使用transformer的编码器结构和MLP，经过大规模数据集预训练之后，模型的分类效果可以超越CNN.

2.U-Net图像分割模型--U-Net文件夹
  项目环境配置： pytorch和cuda版本： 2.0.0+cu118； torchvision版本:0.15.1；其它详见文件夹中的requirements.txt
  论文及详细代码解读：见微信公众号- “Medical AI 研修库”，待更新....
  论文亮点：文章于2015年发表在MICCAI上，首次提出了U型的编码器和解码器结构，并使用跳跃连接实现特征融合，可以有效的提取上下文特征信息实现良好的分割性能。

3.UNETR 3D医学图像分割模型--UNETR文件夹
  ...待更新

               
               
2.GAN网络--GAN文件夹
  本项目第三方库依赖: torch 1.10.0  cuda 11.1 torchvision 0.11  numpy 1.21.6   
  论文及详细代码解读：见微信公众号- “Medical AI 研修库”，待更新....
