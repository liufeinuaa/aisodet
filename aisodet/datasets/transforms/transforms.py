from aisodet.registry import TRANSFORMS
from mmdet.datasets.transforms import RandomFlip as MMdet_RandomFlip
from mmdet.datasets.transforms import Resize as MMdet_Resize
from mmdet.datasets.transforms import Pad as MMdet_Pad
# from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmdet.structures.bbox import autocast_box_type
import mmcv
import numpy as np
from typing import List, Optional, Union
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
import cv2
import pdb

@TRANSFORMS.register_module()
class RandomFlip_sr(MMdet_RandomFlip):
# class RandomFlip_sr(MMCV_RandomFlip):
    """
    用于sr det 架构中，同步hr img 的翻转
    """
    @autocast_box_type()
    def _flip(self, results:dict):
        """Flip images, bounding boxes, and semantic segmentation map.
        and hr img
        """
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].flip_(img_shape, results['flip_direction'])

        # flip masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].flip(
                results['flip_direction'])

        # flip segs
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = mmcv.imflip(
                results['gt_seg_map'], direction=results['flip_direction'])

        # flip hr img
        # pdb.set_trace()
        if results.get('gt_hr_img', None) is not None:
            results['gt_hr_img'] = mmcv.imflip(
                results['gt_hr_img'], direction=results['flip_direction'])

        # record homography matrix for flip
        self._record_homography_matrix(results)




# 保证随机旋转能在aisodet中正常使用
@TRANSFORMS.register_module()
class RandomRotate2(BaseTransform):
    """Random rotate image & bbox & masks. The rotation angle will choice in.

    [-angle_range, angle_range). Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)
    Modified Keys:
    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map
    Added Keys:
    - homography_matrix
    Args:
        prob (float): The probability of whether to rotate or not. Defaults
            to 0.5.
        angle_range (int): The maximum range of rotation angle. The rotation
            angle will lie in [-angle_range, angle_range). Defaults to 180.
        rect_obj_labels (List[int], Optional): A list of labels whose
            corresponding objects are alwags horizontal. If
            results['gt_bboxes_labels'] has any label in ``rect_obj_labels``,
            the rotation angle will only be choiced from [90, 180, -90, -180].
            Defaults to None.
        rotate_type (str): The type of rotate class to use. Defaults to
            "Rotate".
        **rotate_kwargs: Other keyword arguments for the ``rotate_type``.
    """

    def __init__(self,
                 prob: float = 0.5,
                 angle_range: int = 180,
                 rect_obj_labels: Optional[List[int]] = None,
                 rotate_type: str = 'mmrotate.Rotate',
                 **rotate_kwargs) -> None:
        assert 0 < angle_range <= 180
        self.prob = prob
        self.angle_range = angle_range
        self.rect_obj_labels = rect_obj_labels
        self.rotate_cfg = dict(type=rotate_type, **rotate_kwargs)
        self.rotate = TRANSFORMS.build({'rotate_angle': 0, **self.rotate_cfg})
        self.horizontal_angles = [90, 180, -90, -180]

    @cache_randomness
    def _random_angle(self) -> int:
        """Random angle."""
        return self.angle_range * (2 * np.random.rand() - 1)

    @cache_randomness
    def _random_horizontal_angle(self) -> int:
        """Random horizontal angle."""
        return np.random.choice(self.horizontal_angles)

    @cache_randomness
    def _is_rotate(self) -> bool:
        """Randomly decide whether to rotate."""
        return np.random.rand() < self.prob

    def transform(self, results: dict) -> dict:
        """The transform function."""
        if not self._is_rotate():
            return results

        rotate_angle = self._random_angle()
        if self.rect_obj_labels is not None and 'gt_bboxes_labels' in results:
            for label in self.rect_obj_labels:
                if (results['gt_bboxes_labels'] == label).any():
                    rotate_angle = self._random_horizontal_angle()
                    break

        self.rotate.rotate_angle = rotate_angle
        return self.rotate(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'rotate_angle={self.angle_range}, '
        repr_str += f'rect_obj_labels={self.rect_obj_labels}, '
        repr_str += f'rotate_cfg={self.rotate_cfg})'
        return repr_str



@TRANSFORMS.register_module()
class RandomRotate_sr(RandomRotate2):
    """
    适配sr 分支的 随机旋转
    """
    def transform(self, results: dict) -> dict:
        pdb.set_trace()
        return super().transform(results)



@TRANSFORMS.register_module()
class Rotate_sr(BaseTransform):
    """Rotate the images, bboxes, masks and segmentation map by a certain
    angle. Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)
    Modified Keys:
    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map
    Added Keys:
    - homography_matrix
    Args:
        rotate_angle (int): An angle to rotate the image.
        img_border_value (int or float or tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 0.
        mask_border_value (int): The fill value used for masks. Defaults to 0.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Defaults to 255.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 rotate_angle: int,
                 img_border_value: Union[int, float, tuple] = 0,
                 mask_border_value: int = 0,
                 seg_ignore_label: int = 255,
                 interpolation: str = 'bilinear') -> None:
        if isinstance(img_border_value, (float, int)):
            img_border_value = tuple([float(img_border_value)] * 3)
        elif isinstance(img_border_value, tuple):
            assert len(img_border_value) == 3, \
                f'img_border_value as tuple must have 3 elements, ' \
                f'got {len(img_border_value)}.'
            img_border_value = tuple([float(val) for val in img_border_value])
        else:
            raise ValueError(
                'img_border_value must be float or tuple with 3 elements.')
        self.rotate_angle = rotate_angle
        self.img_border_value = img_border_value
        self.mask_border_value = mask_border_value
        self.seg_ignore_label = seg_ignore_label
        self.interpolation = interpolation

    def _get_homography_matrix(self, results: dict) -> np.ndarray:
        """Get the homography matrix for Rotate."""
        img_shape = results['img_shape']
        center = ((img_shape[1] - 1) * 0.5, (img_shape[0] - 1) * 0.5)
        cv2_rotation_matrix = cv2.getRotationMatrix2D(center,
                                                      -self.rotate_angle, 1.0)
        return np.concatenate(
            [cv2_rotation_matrix,
             np.array([0, 0, 1]).reshape((1, 3))],
            dtype=np.float32)

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the geometric transformation."""
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = self.homography_matrix
        else:
            results['homography_matrix'] = self.homography_matrix @ results[
                'homography_matrix']

    def _transform_img(self, results: dict) -> None:
        """Rotate the image."""
        # pdb.set_trace()
        results['img'] = mmcv.imrotate(
            results['img'],
            self.rotate_angle,
            border_value=self.img_border_value,
            interpolation=self.interpolation)

    def _transform_masks(self, results: dict) -> None:
        """Rotate the masks."""
        results['gt_masks'] = results['gt_masks'].rotate(
            results['img_shape'],
            self.rotate_angle,
            border_value=self.mask_border_value,
            interpolation=self.interpolation)

    def _transform_seg(self, results: dict) -> None:
        """Rotate the segmentation map."""
        results['gt_seg_map'] = mmcv.imrotate(
            results['gt_seg_map'],
            self.rotate_angle,
            border_value=self.seg_ignore_label,
            interpolation='nearest')

    def _transform_bboxes(self, results: dict) -> None:
        """Rotate the bboxes."""
        if len(results['gt_bboxes']) == 0:
            return
        img_shape = results['img_shape']
        center = (img_shape[1] * 0.5, img_shape[0] * 0.5)
        results['gt_bboxes'].rotate_(center, self.rotate_angle)
        results['gt_bboxes'].clip_(img_shape)

    def _transform_hr_imgs(self, results: dict):
        """
        增加，旋转hr img
        """
        results['gt_hr_img'] = mmcv.imrotate(
            results['gt_hr_img'],
            self.rotate_angle,
            border_value=self.img_border_value,
            interpolation=self.interpolation
        )


    def _filter_invalid(self, results: dict) -> None:
        """Filter invalid data w.r.t `gt_bboxes`"""
        height, width = results['img_shape']
        if 'gt_bboxes' in results:
            if len(results['gt_bboxes']) == 0:
                return
            bboxes = results['gt_bboxes']
            valid_index = results['gt_bboxes'].is_inside([height,
                                                          width]).numpy()
            results['gt_bboxes'] = bboxes[valid_index]

            # ignore_flags
            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_index]

            # labels
            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                    valid_index]

            # mask fields
            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_index.nonzero()[0]]


    def transform(self, results: dict) -> dict:
        """The transform function."""
        # pdb.set_trace()
        self.homography_matrix = self._get_homography_matrix(results)
        self._record_homography_matrix(results)
        self._transform_img(results)

        # pdb.set_trace()
        if results.get('gt_bboxes', None) is not None:
            self._transform_bboxes(results)
        if results.get('gt_masks', None) is not None:
            self._transform_masks(results)
        if results.get('gt_seg_map', None) is not None:
            self._transform_seg(results)

        # 增加旋转hr img
        if results.get('gt_hr_img', None) is not None:
            self._transform_hr_imgs(results)

        self._filter_invalid(results)
        return results


    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(rotate_angle={self.rotate_angle}, '
        repr_str += f'img_border_value={self.img_border_value}, '
        repr_str += f'mask_border_value={self.mask_border_value}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str




# 增加对 resize 的同步，直接通过加载hr 图像，然后通过 resize 生成 lr 和 lr 对应的 ann

@TRANSFORMS.register_module()
class Resize_sr(MMdet_Resize):
    """Resize images & bbox & seg.

    新增对 hr images 也做 resize 使得其保证为 lr 的 2x 倍数


    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, masks, and seg map are then resized
    with the same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_masks
    - gt_seg_map


    Added Keys:

    - scale
    - scale_factor
    - keep_ratio
    - homography_matrix

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """
    def __init__(self,
                 hr_scale_factor=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.hr_scale_factor = hr_scale_factor

        if hr_scale_factor is None:
            self.hr_scale_facotor = None
        else:
            if isinstance(hr_scale_factor, int):
                self.hr_scale_factor = hr_scale_factor
            else:
                pass


    def _resize_hr_img(self, results: dict) -> None:
        """
        新增的 
        Resize gt hr images with ``results['scale']``.
        
        """
        hr_scale = [lr_scale * self.hr_scale_factor for lr_scale in results['img_shape']]
        # results['scale'] * self.hr_scale_factor
        # pdb.set_trace()
        results['hr_scale'] = tuple(hr_scale)

        if results.get('gt_hr_img', None) is not None:
            
            img, scale_factor = mmcv.imrescale(
                    results['gt_hr_img'],
                    results['hr_scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future

            # 填充下 0 值 是的 处理后的 hr img 的size 确实为 lr img 的2倍
            if results['img_shape'][0]*2 != img.shape[0]:
                # pdb.set_trace()
                img = mmcv.impad(img, shape=(results['img_shape'][0]*2, img.shape[1]))
            
            if results['img_shape'][1]*2 != img.shape[1]:
                # pdb.set_trace()
                img = mmcv.impad(img, shape=(img.shape[0], results['img_shape'][1]*2))


            new_h, new_w = img.shape[:2]
            h, w = results['gt_hr_img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h

            results['gt_hr_img'] = img
            results['gt_hr_img_shape'] = img.shape[:2]
            results['hr_scale_factor'] = (w_scale, h_scale)

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        if self.scale:
            results['scale'] = self.scale
            results['hr_scale_factor'] = self.hr_scale_factor
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)
            results['hr_scale_factor'] = self.hr_scale_factor
            
        # pdb.set_trace()
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._record_homography_matrix(results)
        self._resize_hr_img(results)
        # pdb.set_trace()
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class Resize_sr_v2(MMdet_Resize):
    """Resize images & bbox & seg.

    新增对 hr images 也做 resize 使得其保证为 lr 的 2x 倍数

    """
    def __init__(self,
                 hr_scale_factor=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.hr_scale_factor = hr_scale_factor


    def _resize_hr_img(self, results: dict) -> None:
        """
        新增的 
        Resize gt hr images with ``results['scale']``.
        
        """
        hr_scale = [lr_scale * self.hr_scale_factor for lr_scale in results['img_shape']]
        results['hr_scale'] = tuple(hr_scale)

        if results.get('gt_hr_img', None) is not None:
            
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['gt_hr_img'],
                    results['hr_scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['gt_hr_img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['gt_hr_img'],
                    results['hr_scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            
            results['gt_hr_img'] = img
            results['gt_hr_img_shape'] = img.shape[:2]
            results['hr_scale_factor'] = (w_scale, h_scale)
    



    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        if self.scale:
            results['scale'] = self.scale
            results['hr_scale_factor'] = self.hr_scale_factor
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)
            results['hr_scale_factor'] = self.hr_scale_factor
            
        # pdb.set_trace()
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._record_homography_matrix(results)
        self._resize_hr_img(results)
        # pdb.set_trace()
        return results





# 增加 对hr填充的 操作
@TRANSFORMS.register_module()
class Pad_sr(MMdet_Pad):
    """Pad the image & segmentation map.

    There are three padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number. and (3)pad to square. Also,
    pad to square and pad to the minimum size can be used as the same time.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_masks
    - gt_seg_map

    Added Keys:

    - pad_shape
    - pad_fixed_size
    - pad_size_divisor

    Args:
        size (tuple, optional): Fixed padding size.
            Expected padding shape (width, height). Defaults to None.
        size_divisor (int, optional): The divisor of padded size. Defaults to
            None.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Defaults to False.
        pad_val (Number | dict[str, Number], optional) - Padding value for if
            the pad_mode is "constant".  If it is a single number, the value
            to pad the image is the number and to pad the semantic
            segmentation map is 255. If it is a dict, it should have the
            following keys:

            - img: The value to pad the image.
            - seg: The value to pad the semantic segmentation map.
            Defaults to dict(img=0, seg=255).
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Defaults to 'constant'.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]
    """
    def _pad_hr_img(self, results):
        """
        新增的方法，对 hr img 也进行填充，保证 img 与 hr img 都为 32 的倍数才行, 并且 hr img 达到的倍数刚好为 lr 的两倍
        """
        pad_val = self.pad_val.get('img', 0)

        lr_size = (results['img'].shape[0], results['img'].shape[1])

        pad_h = int(lr_size[0] * 2)
        pad_w = int(lr_size[1] * 2)
        size = (pad_h, pad_w)
        padded_img = mmcv.impad(
            results['gt_hr_img'],
            shape=size,
            pad_val=pad_val,
            padding_mode=self.padding_mode)

        results['gt_hr_img'] = padded_img
        results['gt_hr_pad_shape'] = padded_img.shape
        results['gt_hr_pad_fixed_size'] = self.size
        results['gt_hr_pad_size_divisor'] = self.size_divisor
        results['pad_gt_hr_img_shape'] = padded_img.shape[:2]

    def transform(self, results: dict) -> dict:
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_seg(results)
        self._pad_masks(results)
        # pdb.set_trace()
        self._pad_hr_img(results)
        # pdb.set_trace()
        return results