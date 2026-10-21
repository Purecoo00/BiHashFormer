# Implementation details

## Architecture

| Component | Configuration |
| --- | --- |
| Stage depths | `[2,2,6,2]` |
| Stage channels | `[96,192,384,768]` |
| Attention heads | `[3,6,12,24]` |
| Window size | 21 |
| RoI count | `ceil(predicted_ratio * number_of_windows)` |
| Ratio predictor | Two-layer MLP; adaptive pooling sizes `[32,16,8,4]` |
| Selection loss | Foreground-ratio squared error plus `0.1 * predicted_ratio` |
| Hamming code | 32 bits; separate linear Q/K projectors |
| Key selection | Nearest half of the keys for each query |
| Prototype count | 3 |
| Retrieval radius | 4 |
| Prototype losses | Nearest-center distance and separation margin `2r+1` |
| Compensation | Self-attention and FFN on retrieved non-RoI tokens |
| Detector | Standard DINO head, encoder and decoder |

Patch embedding/merging and multiscale outputs use MMDetection's Swin interface. Windows alternate between regular and shifted partitions. Padding and cyclic-shift boundaries are masked in both branches. Branch outputs are scattered back into their original positions; unselected non-RoI tokens retain their input features.

## Routing and training conventions

- The stage-level ratio predictor is shared by the Hash Blocks in that stage. Each block recomputes its window scores. The foreground-ratio target is the exact union area of ground-truth boxes clipped to the image bounds.
- Hash routing forms one code per complete token Q/K vector. Selected key indices are shared across attention heads; attention uses the original continuous Q/K/V features.
- Codes are strictly binary, with zero-valued latents mapped to bit 0. The STE derivative with respect to each latent is 1/2. Prototype latents start at +/-1 and represent disjoint blocks of foreground bits.
- The Hamming forward path uses packed XOR/popcount. During training, an algebraic continuous-bit distance supplies its backward surrogate. A zero-valued selected-affinity STE term in attention logits passes gradients through hard key routing while preserving the attention forward values. Set `routing_ste=False` to disable it.
- Window scores use an identity-forward STE gain to train the scorer through hard top-k window routing.
- A non-RoI token is retained if its Hamming distance to at least one prototype is within the radius: `min_i distance(token, center_i) <= r`.
- The prototype objective minimizes normalized distance to the nearest center and penalizes center pairs closer than `2r+1`. Prototype training uses keys selected in RoI windows.
- HashMiner reuses the compression branch's K transform/projector with gradient tracking disabled on that branch. Those parameters remain trainable through the compression branch.
- `BiHashDINO` adds `loss_select` and `loss_proto` to the detector's ordinary losses without changing its head, encoder or decoder.

## Runtime

Query/window chunking and activation checkpointing bound intermediate memory. This implementation uses PyTorch operators and has no custom fused sparse-attention CUDA kernel. The synthetic validation described in [validation.md](validation.md) checks numerical behavior and training integration; it does not establish end-to-end benchmark accuracy or runtime.
