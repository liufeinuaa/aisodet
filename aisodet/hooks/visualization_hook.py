import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
from mmengine.fileio import FileClient
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

# from mmdet.registry import HOOKS
from aisodet.registry import HOOKS

from mmdet.structures import DetDataSample
from mmdet.engine.hooks import DetVisualizationHook
from mmengine import MessageHub
import pdb
import numpy as np

@HOOKS.register_module()
class rotDetVisualizationHook(DetVisualizationHook):
    """
    直接继承自 mmedet中的 hook

    对 我增加的 方法做 特征图可视化 操作
    
    """
    def __init__(self, 
                 out_heatmaps=False,
                 out_p2_only=False,
                 **kwargs):
        self.out_heatmaps = out_heatmaps
        self.out_p2_only = out_p2_only

        super().__init__(**kwargs)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:
        """
        主要在这边进行定制化
        
        Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        # pdb.set_trace()
        message_hub = MessageHub.get_current_instance()
        # pdb.set_trace()
        heatmaps = message_hub.get_info('heatmaps')
        
        # —————————— 保留下面的 代码 ————————————————————
        if self.draw is False:
            return

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.img_path
            img_bytes = self.file_client.get(img_path)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            out_file = None
            if self.test_out_dir is not None:
                out_file = osp.basename(img_path)
                out_file = osp.join(self.test_out_dir, out_file)

            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'test_img',
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=out_file,
                step=self._test_index)

            
            # pdb.set_trace()
            # 新增的输出 heatmap 的部分
            if out_file is not None:
                prex_path = osp.splitext(out_file)[0]
                if self.out_heatmaps:
                    # pdb.set_trace()
                    drawn_c2 = heatmaps[0]  # left 为 backbone 后 , right 为 upSB 后
                    # mmcv.imwrite(drawn_img, 'ts_heatmaps.png')
                    mmcv.imwrite(drawn_c2, prex_path+'_c2.png')

                    drawn_c2up = heatmaps[1]  # left 为 backbone 后 , right 为 upSB 后
                    # mmcv.imwrite(drawn_img, 'ts_heatmaps.png')
                    mmcv.imwrite(drawn_c2up, prex_path+'_c2up.png')

                    drawn_p2 = heatmaps[2] # left 为 backbone 后 , right 为 upSB 后
                    # mmcv.imwrite(drawn_img, 'ts_heatmaps.png')
                    mmcv.imwrite(drawn_p2, prex_path+'_p2.png')

                    drawn_sr = heatmaps[3]  # left 为 backbone 后 , right 为 upSB 后
                    # mmcv.imwrite(drawn_img, 'ts_heatmaps.png')
                    mmcv.imwrite(drawn_sr, prex_path+'_sr.png')
                elif self.out_p2_only:
                    drawn_p2 = heatmaps[0]
                    mmcv.imwrite(drawn_p2, prex_path+'_p2.png')
                else:
                    drawn_img = np.concatenate(heatmaps, axis=1)  # left 为 backbone 后 , right 为 upSB 后
                    # mmcv.imwrite(drawn_img, 'ts_heatmaps.png')
                    mmcv.imwrite(drawn_img, prex_path+'_heatmaps.png')