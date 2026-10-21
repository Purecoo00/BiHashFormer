"""Build the published configuration, take one synthetic optimizer step, predict.

This validates code execution; it does not measure detection accuracy.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dino/dino_bihashformer_t_8xb1_36e_panda.py")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    register_all_modules()
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).to(args.device)
    model.init_weights()
    model.train()
    size = args.size
    sample = DetDataSample(
        metainfo=dict(
            img_shape=(size, size),
            ori_shape=(size, size),
            batch_input_shape=(size, size),
            scale_factor=(1.0, 1.0),
        )
    )
    sample.gt_instances = InstanceData(
        bboxes=torch.tensor(
            [[8.0, 9.0, 40.0, 90.0], [100.0, 30.0, 140.0, 80.0]], device=args.device
        ),
        labels=torch.zeros(2, dtype=torch.long, device=args.device),
    )
    image = torch.randn(1, 3, size, size, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    losses = model.loss(image, [sample])
    total = sum(
        v.mean() if torch.is_tensor(v) else sum(t.mean() for t in v)
        for k, v in losses.items()
        if "loss" in k
    )
    if not torch.isfinite(total):
        raise RuntimeError("Non-finite training loss")
    total.backward()
    gradients = {
        "roi_predictor": model.backbone.stages[0].ratio_predictor[0].weight.grad,
        "window_score": model.backbone.stages[0].blocks[0].score_net.weight.grad,
        "query_projector": model.backbone.stages[0]
        .blocks[0]
        .compression.query_projector.weight.grad,
        "key_projector": model.backbone.stages[0].blocks[0].compression.key_projector.weight.grad,
        "prototype": model.backbone.stages[0].blocks[0].compression.prototypes.latents.grad,
    }
    for name, grad in gradients.items():
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
            raise RuntimeError(f"Missing/invalid gradient: {name}")
    print("Finite training loss:", float(total.detach()))
    print("Nonzero finite gradients:", ", ".join(gradients))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        predictions = model.predict(image, [sample])
    assert len(predictions) == 1 and torch.isfinite(predictions[0].pred_instances.bboxes).all()
    print("Optimizer step and DINO prediction passed.")
    if args.device.startswith("cuda"):
        print("Peak GPU allocated MiB:", round(torch.cuda.max_memory_allocated() / 1024**2, 1))


if __name__ == "__main__":
    main()
