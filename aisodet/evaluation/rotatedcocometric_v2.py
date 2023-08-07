# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, Optional, Sequence
import torch
import cv2
import numpy as np
import pycocotools.mask as maskUtils
from mmcv.ops import box_iou_rotated
from mmdet.datasets.api_wrappers import COCO
from mmdet.evaluation import CocoMetric
from mmengine import MMLogger
from mmengine.fileio import dump, load
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

# from mmrotate.registry import METRICS
from aisodet.registry import METRICS
from mmrotate.structures.bbox import RotatedBoxes
from mmrotate.evaluation import RotatedCocoMetric
from mmdet.structures.mask import encode_mask_results
import pdb

from aitodpycocotools.cocoeval import COCOeval as aitodCOCOeval


@METRICS.register_module()
class RotatedCocoMetricv2(RotatedCocoMetric):
    """
    为了测试小目标下的检测精度, 启用这边的 coco 类型的评估方法
    """
    def __init__(self, 
                use_aitodeval=False, 
                **kwargs):
        super().__init__(**kwargs)
        self.use_aitodeval = use_aitodeval
        


    # 这里好像有问题, 需要调整下
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                # pdb.set_trace()
                # TODO: Need to refactor to support LoadAnnotations
                # assert 'instances' in data_sample, \
                #     'ground truth is required for evaluation when ' \
                #     '`ann_file` is not provided'
                # gt['anns'] = data_sample['instances']

                assert 'gt_instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['gt_instances']
            # add converted result to the results list
            self.results.append((gt, result))

    # 这个方法有问题, 需要调整下
    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        categories = [
            dict(id=id, name=name)
            for id, name in enumerate(self.dataset_meta['classes'])
        ]
        image_infos = []
        annotations = []

        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                file_name='')
            image_infos.append(image_info)

            # for ann in gt_dict['anns']:
            # pdb.set_trace()
            for idx in range(len(gt_dict['anns']['labels'])):
                # pdb.set_trace()
                label = gt_dict['anns']['labels'][idx].cpu().numpy()
                coco_bbox = gt_dict['anns']['bboxes'][idx].cpu().numpy()
                # coco_bbox = qbox2rbox_list(bbox)
                # label = ann['labels']


                annotation = dict(
                    id=len(annotations) +
                    1,  # coco api requires id starts with 1
                    image_id=img_id,
                    bbox=coco_bbox,
                    # iscrowd=ann.get('ignore_flag', 0),
                    iscrowd=0,
                    category_id=int(label),
                    area=coco_bbox[2] * coco_bbox[3])
                
                # if ann.get('mask', None):
                #     mask = ann['mask']
                #     # area = mask_util.area(mask)
                #     if isinstance(mask, dict) and isinstance(
                #             mask['counts'], bytes):
                #         mask['counts'] = mask['counts'].decode()
                #     annotation['segmentation'] = mask
                    # annotation['area'] = float(area)
                annotations.append(annotation)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmdet CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path)
        return converted_json_path


    # 这里尝试使用 对tiny object 的评估
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            # TODO: May refactor fast_eval_recall to an independent metric?
            # fast eval recall
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    preds, self.proposal_nums, self.iou_thrs, logger=logger)
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg)
                continue

            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error(
                    'The testing results of the whole dataset is empty.')
                break
            
            # pdb.set_trace()
            if self.use_aitodeval:
                coco_eval = RotatedCocoEval_tiny(self._coco_api, coco_dt, iou_type)
                # mapping of cocoEval.stats
                coco_metric_names = {
                    'mAP': 0,
                    'mAP_25': 1,
                    'mAP_50': 2,
                    'mAP_75': 3,
                    'mAP_vt': 4,
                    'mAP_t': 5,
                    'mAP_s': 6,
                    'mAP_m': 7,
                    'AR@100': 8,
                    'AR@300': 9,
                    'AR@1500': 10,
                    'AR_vt@1500': 11,
                    'AR_t@1500': 12,
                    'AR_s@1500': 13,
                    'AR_m@1500': 14,
                    'oLRP': 15,
                    'oLRP_Localisation': 16,
                    'oLRP_false_positive': 17,
                    'oLRP_false_negative': 18
                }
            else:
                coco_eval = RotatedCocoEval(self._coco_api, coco_dt, iou_type)
                # mapping of cocoEval.stats
                coco_metric_names = {
                    'mAP': 0,
                    'mAP_50': 1,
                    'mAP_75': 2,
                    'mAP_s': 3,
                    'mAP_m': 4,
                    'mAP_l': 5,
                    'AR@100': 6,
                    'AR@300': 7,
                    'AR@1000': 8,
                    'AR_s@1000': 9,
                    'AR_m@1000': 10,
                    'AR_l@1000': 11
                }

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            
            # elif metric == 'aitod': # 新增的使用 aitod 的借口,能评估极小目标
            
            else: # 就是使用的 metric='bbox'
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if self.classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = coco_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, cat_id in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self._coco_api.loadCats(cat_id)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{round(ap, 3)}'))
                        eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info('\n' + table.table)

                if metric_items is None:
                    if self.use_aitodeval:
                        metric_items = [
                            'mAP', 'mAP_50', 'mAP_75', 'mAP_vt', 'mAP_t', 'mAP_s', 'mAP_m', 'oLRP', 'oLRP_Localisation', 'oLRP_false_positive', 'oLRP_false_negative'
                        ]
                    else:
                        metric_items = [
                            'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                        ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        
        # pdb.set_trace()
        return eval_results



class RotatedCocoEval_tiny(aitodCOCOeval):
    """This is a wrapper to support Rotated Box Eval."""

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            # For segm, its same with original COCOeval
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
            # compute iou between each dt and gt region
            iscrowd = [int(o['iscrowd']) for o in gt]
            ious = maskUtils.iou(d, g, iscrowd)
        elif p.iouType == 'bbox':
            # Modified for Rotated Box
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
            # Convert List[List[float]] to Tensor for iou compute
            g = RotatedBoxes(g).tensor
            d = RotatedBoxes(d).tensor
            ious = box_iou_rotated(d, g)
        else:
            raise Exception('unknown iouType for iou computation')

        return ious


class RotatedCocoEval(COCOeval):
    """This is a wrapper to support Rotated Box Eval."""

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            # For segm, its same with original COCOeval
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
            # compute iou between each dt and gt region
            iscrowd = [int(o['iscrowd']) for o in gt]
            ious = maskUtils.iou(d, g, iscrowd)
        elif p.iouType == 'bbox':
            # Modified for Rotated Box
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
            # Convert List[List[float]] to Tensor for iou compute
            g = RotatedBoxes(g).tensor
            d = RotatedBoxes(d).tensor
            ious = box_iou_rotated(d, g)
        else:
            raise Exception('unknown iouType for iou computation')

        return ious
