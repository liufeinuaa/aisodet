"""
这里打算使用mmcv自带的resize方法，将img 连同 ann 一起resize成512x512（down 2x， 原始尺寸为1024x1024）

版本2：
增加对dota test 数据集的处理
"""

from mmrotate.datasets import DOTADataset
from mmcv.image import imwrite
import os.path as osp
import codecs
import tqdm
import os
import argparse
import pdb


CLSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter')
label_map = {i:c for i, c in enumerate(CLSES)}


def add_parser(parser):
    """Add arguments."""
    parser.add_argument(
        '--hr-data-root',
        type=str,
        default='data2/split_1024_dota_mini4',
        help='就是原始hr图像数据集的路径'
    )
    parser.add_argument(
        '--sub-file',
        type=str,
        default='trainval',
        help='具体处理的子数据集（val？ or trainval）， 增加对test数据集的处理（没有ann）'
    )
    parser.add_argument(
        '--out-path',
        type=str,
        default='./data2/split_lr512_dota_mini4',
        help='2x下采样完成后数据集的输出路径'
    )
    parser.add_argument(
        '--down_func',
        type=str,
        default='bicubic',
        help='降低分辨率具体使用的方法（bicubic or bilinear 等）'
    )

def parse_args():
    parser = argparse.ArgumentParser(description='down resolution of datasets')
    add_parser(parser)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # pdb.set_trace()
    down_2x_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', 
        with_bbox=True, box_type='qbox'),
    # dict(type='mmdet.Resize',  # 使用默认的双线性下采样
    #     scale=(512, 512), keep_ratio=True), # 这里就完成了2倍下采样
    dict(type='mmdet.Resize',  # 使用双三次下采样
        scale=(512, 512), keep_ratio=True,
        interpolation=args.down_func), # 换用mmcv的实现就不会有伪影存在
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
    ]

    # datasets = DOTADataset(
    #     # 使用的数据集
    #     data_root=args.hr_data_root,
        
    #     # 创建2x下采样后的trainval
    #     ann_file=f'{args.sub_file}/annfiles/',
    #     data_prefix=dict(img_path=f'{args.sub_file}/images/'),
        
    #     img_shape=(1024, 1024),
    #     filter_cfg=dict(filter_empty_gt=True),
    #     pipeline=down_2x_pipeline,
    # )

    # 针对test 数据集的处理
    test_down_2x_pipeline = [
        dict(type='mmdet.LoadImageFromFile'),
        dict(type='mmdet.Resize',  # 使用双三次下采样
        scale=(512, 512), keep_ratio=True,
        interpolation=args.down_func), # 换用mmcv的实现就不会有伪影存在
        dict(type='mmdet.Pad', size=(512, 512),
        pad_val=dict(img=(114, 114, 114))),
        dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
    ]

    # pdb.set_trace()
    if args.sub_file == 'test':
        # pdb.set_trace()
        datasets = DOTADataset(
            # 使用的数据集
            data_root=args.hr_data_root,
            
            # 创建2x下采样后的trainval
            # ann_file=f'{args.sub_file}/annfiles/',
            data_prefix=dict(img_path=f'{args.sub_file}/images/'),
            
            img_shape=(1024, 1024),
            filter_cfg=dict(filter_empty_gt=True), # 在这里启用了过滤掉没有gt 的 imgs
            pipeline=test_down_2x_pipeline,
            )
    else:
        datasets = DOTADataset(
            # 使用的数据集
            data_root=args.hr_data_root,
            
            # 创建2x下采样后的trainval
            ann_file=f'{args.sub_file}/annfiles/',
            data_prefix=dict(img_path=f'{args.sub_file}/images/'),
            
            img_shape=(1024, 1024),
            filter_cfg=dict(filter_empty_gt=True), # 在这里启用了过滤掉没有gt 的 imgs
            test_mode=True,
            pipeline=down_2x_pipeline,
            )

    # 低分辨率2x下采样，数据集输出的路径
    lr_out_dir = args.out_path
    
    # 将两倍下采样的结果写入文件夹中
    with tqdm.tqdm(total=len(datasets)) as pbar:
        pbar.set_description('trans orig_img to lr_2x_img Processing:')
        # pdb.set_trace()
        for i, data in enumerate(datasets):
            img = data['inputs'] # 得到的img就是下采样后的
            img_np = img.permute(1, 2, 0).numpy() # 将图片由tensor转换成opencv能保存的np array的格式

            # 保存img到指定的文件夹中
            img_name = data['data_samples'].img_path.split('/')[-1]
            # pdb.set_trace()
            lr_out_path = lr_out_dir + f'/{args.sub_file}/images/' + img_name
            imwrite(img_np, lr_out_path)

            # pdb.set_trace()
            if args.sub_file == 'test':
                # pass
                pbar.set_description(f"{img_name}'s img save done")
            else:
                # 保存label到指定的文件夹中
                anns = data['data_samples'].gt_instances
                labels = anns.labels
                # pdb.set_trace()
                labels_name = [label_map[int(label)] for label in labels]
                qbboxes = anns.bboxes
                qbboxes_np = qbboxes.numpy()
                # pdb.set_trace()

                txtname = f"{img_name.split('.')[0]}.txt"
                ann_out_path = lr_out_dir + f'/{args.sub_file}/annfiles/'
                
                if not osp.exists(ann_out_path):
                    os.makedirs(ann_out_path)
                
                txtfile = osp.join(ann_out_path, txtname)

                # 写入txt文件中
                with codecs.open(txtfile, 'w', 'utf-8') as f_out:
                    for i in range(len(qbboxes_np)):
                        qbbox_str = ''
                        for j in range(8):
                            tmp = str(qbboxes_np[i][j])
                            if j == 0:
                                qbbox_str = tmp
                            else:
                                qbbox_str = f"{qbbox_str} {tmp}"
                        # pdb.set_trace()
                        # 组合成str
                        line = f"{qbbox_str} {labels_name[i]} 0\n"
                        # pdb.set_trace()
                        f_out.writelines(line)
            
                pbar.set_description(f"{img_name}'s ann save done")
            
            pbar.update(1)



if __name__ == '__main__':
    main()

    """
    # 对dota trainval 下的文件执行下采样（默认情况）
    python tools/sr/get_lr_2x_dota2.py --hr-data-root data2/split_1024_dota1_0 --out-path './data2/split_lr512_dota/'

    # 对dota 的 val文件夹下的文件执行下采样---ok
    python tools/sr/get_lr_2x_dota2.py --hr-data-root data2/split_1024_dota1_0 --sub-file val --out-path './data2/split_lr512_dota/'

    # 对dota test下的文件执行下采样
    python tools/sr/get_lr_2x_dota2.py --hr-data-root data2/split_1024_dota1_0 --sub-file test --out-path './data2/split_lr512_dota/'

    """
