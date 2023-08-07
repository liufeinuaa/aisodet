这里这个文件夹中的脚本，是用于生成用于sr网络训练中的，下采样2x的lr images （低分辨率图像）

- 主要用于dota v1 数据集中，对应的脚本文件为 get_lr_2x_dota2.py
    按照文件 main 部分的注释代码可以生成对应的 lr images

- 而aod 使用 在图像加载过程中的 pipeline 来生成对应的 lr


