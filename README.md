# BiHashFormer

**BiHashFormer: Hash-Driven Dual-Branch Transformer for Efficient Object Detection in HRW Shots**  
NeurIPS 2026  
Jingchen Huang*, Wenxi Li*, Chenyang Lyu, Moran Liu, Haozhe Lin, Yuchen Guo†  
*Equal contribution. †Corresponding author.

[Paper](https://openreview.net/forum?id=VT9aN17Fbk) · [Implementation notes](docs/implementation.md) · [Data preparation](docs/data.md)

BiHashFormer processes high-resolution, wide-field images with two branches. A learned RoI Selector sends promising windows to the **compression branch**, where each query attends to the nearest half of the keys in a learned Hamming space. The **compensation branch** uses three foreground prototypes to retrieve tokens from the remaining windows, then applies self-attention to the retrieved tokens.

This repository provides the PyTorch implementation of BiHashFormer, including the backbone, DINO integration, training configurations and PANDA inference/evaluation tools. Trained weights are not included.

## Installation

The validation environment uses Python 3.10, PyTorch 2.1.0/CUDA 11.8, MMCV 2.1.0, MMEngine 0.10.7 and MMDetection 3.3.0.

```bash
conda create -n bihashformer python=3.10 -y
conda activate bihashformer
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install -r requirements.txt
pip install -e . --no-deps
```

Use the MMCV wheel matching your PyTorch/CUDA build; avoid installing `mmcv-lite`, which lacks the deformable attention operator required by DINO.

## Training and evaluation

Prepare COCO-format PANDA patches as described in [data.md](docs/data.md). The default configuration uses 36 epochs and one image per GPU; the launcher below uses eight GPUs.

```bash
bash scripts/dist_train.sh configs/dino/dino_bihashformer_t_8xb1_36e_panda.py 8
bash scripts/dist_test.sh configs/dino/dino_bihashformer_t_8xb1_36e_panda.py path/to/checkpoint.pth 8
```

For one GPU:

```bash
python tools/train.py configs/dino/dino_bihashformer_t_8xb1_36e_panda.py
python tools/test.py configs/dino/dino_bihashformer_t_8xb1_36e_panda.py path/to/checkpoint.pth
```

The standard test configuration evaluates **patch-level COCO metrics**. PANDA's full-frame metrics require remapping and merging tile predictions and the PANDA-specific area ranges; do not equate patch metrics with the main-paper table. `tools/infer_tiles.py` provides tiled full-frame inference and COCO-format prediction export. `tools/evaluate_panda.py` applies the PANDA evaluator adopted from SparseFormer.

## Validation

```bash
python -m pytest tests -q
python tools/smoke_test.py --device cuda:0
```

The synthetic smoke test runs the DINO configuration for one optimizer step and inference. It does not train a useful detector or establish the paper's accuracy.

## Repository structure

```text
bihash_former/models/backbones/bihashformer.py  # RoI Selector, HashAttention, HashMiner, backbone
bihash_former/models/detectors/bihash_dino.py   # DINO + selection/prototype losses
configs/                                     # portable dataset/model/training configuration
tools/                                       # train, test, tiled inference, PANDA evaluation
scripts/                                     # distributed launchers
tests/                                       # numerical and routing checks
docs/                                        # implementation details and data layout
```

## Citation

```bibtex
@inproceedings{huang2026bihashformer,
  title={BiHashFormer: Hash-Driven Dual-Branch Transformer for Efficient Object Detection in HRW Shots},
  author={Huang, Jingchen and Li, Wenxi and Lyu, Chenyang and Liu, Moran and Lin, Haozhe and Guo, Yuchen},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Acknowledgments and license

The repository organization follows [liwenxi/SparseFormer](https://github.com/liwenxi/SparseFormer). The backbone reuses [MMDetection](https://github.com/open-mmlab/mmdetection)'s Swin patch embedding, patch merging and feature interface, and the detector uses MMDetection's DINO head, encoder and decoder. PANDA evaluation is adapted from SparseFormer. Code is distributed under the Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
