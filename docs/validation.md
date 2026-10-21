# Validation, 2026-10-21

Environment: Python 3.10.19; PyTorch 2.1.0+cu118; torchvision 0.16.0+cu118; MMCV 2.1.0; MMEngine 0.10.7; MMDetection 3.3.0; NumPy 1.26.4; Shapely 2.1.2.

- 10 model tests passed: exact packed Hamming distances at 16/32/64 bits, binary zero handling, separated prototype initialization, query-specific half-key selection, boundary masking, equivalence to dense attention at full selection, train/eval routing agreement, projector/prototype gradients, compensation key-projector freezing, exact foreground union area, and odd-size backbone forward/backward with checkpointing.
- 2 tool tests passed: edge-covering tiling and perfect-detection PANDA evaluation with the intended area ranges and detection limit.
- All five command-line entrypoints respond to `--help`.
- The standard MMEngine Runner completed a synthetic training iteration with the released model, optimizer and registry configuration.
- On an NVIDIA GeForce RTX 4090, the full released DINO configuration completed a synthetic 224x224 training forward/backward, AdamW optimizer step and detector prediction. RoI predictor, window scorer, Q/K hash projectors and prototypes all received finite nonzero gradients.

These checks establish executable model/training integration. They do not establish 36-epoch detection accuracy or the paper's runtime/FLOP measurements. No trained model weights are included.
