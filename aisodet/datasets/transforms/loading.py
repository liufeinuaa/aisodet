"""
将加载hr img 的操作放在load ann中
"""

from mmdet.datasets.transforms import LoadAnnotations as MMdet_LoadAnnotations
from aisodet.registry import TRANSFORMS
import mmcv
import numpy as np
import pdb
import mmengine

@TRANSFORMS.register_module()
class LoadAnnotations_sr(MMdet_LoadAnnotations):
    """
    将加载hr img 的操作放在load ann中, 作为个像素标签的形式
    """
    def __init__(self, 
                 with_hr_img:bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.with_hr_img = with_hr_img
        self.color_type = 'color'
        self.ignore_empty=False
        self.to_float32=False

        # pdb.set_trace()
        # mmengine.fileio.get()


    def _load_hr_imgs(self, results:dict):
        """Private function to load hr images.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        filename = results['hr_path']

        try:
            if float(mmengine.__version__[:3]) > 0.3: # 判断mmengine 的版本号，是否大于 0.3版
                img_bytes = mmengine.fileio.get(filename) # 尝试新版的方法
            else:
                img_bytes = self.file_client.get(filename) # mmengine 0.0.4 版本中取消掉了这个接口方法, 并且mmcv版本不能大雨rc3 。。。tmd
            
            # pdb.set_trace()
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            img = img.astype(np.float32)

        results['gt_hr_img'] = img
        results['gt_hr_img_shape'] = img.shape[:2]
        return results


    def transform(self, results: dict) -> dict:
        # pdb.set_trace()
        super().transform(results)
        if self.with_hr_img:
            self._load_hr_imgs(results)
        
        # pdb.set_trace()
        return results