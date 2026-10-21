# Copyright (c) 2026 BiHashFormer authors. SPDX-License-Identifier: Apache-2.0
"""Standard DINO detector with the backbone's two auxiliary training losses."""
import torch
from shapely.geometry import box
from shapely.ops import unary_union

from mmdet.models.detectors import DINO
from mmdet.registry import MODELS


def foreground_ratio(boxes, image_shape):
    """Exact clipped union area, rather than the sum of overlapping boxes."""
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError('Image dimensions must be positive')
    if hasattr(boxes, 'tensor'):
        boxes = boxes.tensor
    rectangles = []
    for x1, y1, x2, y2 in boxes.detach().cpu().tolist():
        x1, x2 = max(0., x1), min(float(width), x2)
        y1, y2 = max(0., y1), min(float(height), y2)
        if x2 > x1 and y2 > y1:
            rectangles.append(box(x1, y1, x2, y2))
    return unary_union(rectangles).area / (height * width) if rectangles else 0.


@MODELS.register_module()
class BiHashDINO(DINO):
    """DINO head/encoder/decoder remain unchanged."""
    def loss(self, batch_inputs, batch_data_samples):
        losses = super().loss(batch_inputs, batch_data_samples)
        ratios = batch_inputs.new_tensor([
            foreground_ratio(sample.gt_instances.bboxes, sample.img_shape)
            for sample in batch_data_samples
        ])
        losses.update(self.backbone.auxiliary_losses(ratios))
        return losses
