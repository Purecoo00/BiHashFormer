import pytest
import torch
from bihash_former.models.backbones.bihashformer import (
    BiHashFormer, HashAttention, HashMiner, PrototypeCodes,
    binary_code_ste, pairwise_hamming)
from bihash_former.models.detectors.bihash_dino import foreground_ratio


def test_binary_codes_and_separated_initial_prototypes():
    x = torch.tensor([-1., 0., 1.], requires_grad=True)
    y = binary_code_ste(x)
    assert y.tolist() == [0., 0., 1.]
    y.sum().backward()
    assert x.grad.tolist() == [.5, .5, .5]
    p = PrototypeCodes(32, 3, 4)
    codes = p.bits()
    assert codes.unique().tolist() == [0., 1.]
    distances = pairwise_hamming(codes, codes)
    assert distances[~torch.eye(3, dtype=torch.bool)].min() >= 9


@pytest.mark.parametrize('bits', [16, 32, 64])
def test_packed_hamming_equals_explicit_xor(bits):
    torch.manual_seed(3)
    a = torch.randint(2, (2, 7, bits)).float().requires_grad_()
    b = torch.randint(2, (2, 5, bits)).float().requires_grad_()
    expected = (a[:, :, None].detach().bool() ^ b[:, None].detach().bool()).sum(-1)
    got = pairwise_hamming(a, b)
    assert torch.equal(got, expected.float())
    got.sum().backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_nearest_half_is_query_specific_and_respects_regions():
    attention = HashAttention(8, 2, window_size=2, num_bits=4)
    bits = torch.tensor([[[0., 0., 0., 0.], [0., 0., 0., 1.],
                          [1., 1., 1., 0.], [1., 1., 1., 1.]]])
    (d, i), _ = attention.route(bits, bits, torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 4, dtype=torch.long))
    assert i.shape == (1, 4, 2)
    assert set(i[0, 0].tolist()) == {0, 1}
    assert set(i[0, 3].tolist()) == {2, 3}
    (_, i), _ = attention.route(bits, bits, torch.ones(1, 4, dtype=torch.bool), torch.tensor([[0, 0, 1, 1]]))
    assert all(set(i[0, q].tolist()) <= ({0, 1} if q < 2 else {2, 3}) for q in range(4))


def test_full_selection_matches_dense_attention_and_train_eval_agree():
    torch.manual_seed(4)
    a = HashAttention(8, 2, window_size=2, num_bits=4, kv_ratio=1., query_chunk_size=3)
    x = torch.randn(2, 4, 8)
    valid, regions = torch.ones(2, 4, dtype=torch.bool), torch.zeros(2, 4, dtype=torch.long)
    q, k, v = [t.reshape(2, 4, 2, 4).transpose(1, 2) for t in a.qkv(x).chunk(3, -1)]
    bias = a.relative_bias[a.relative_index].permute(2, 0, 1)
    weights = ((q @ k.transpose(-1, -2)) * a.scale + bias).softmax(-1)
    dense = a.proj((weights @ v).transpose(1, 2).reshape(2, 4, 8))
    a.train()
    train, _ = a(x, valid, regions)
    a.eval()
    evaluation, _ = a(x, valid, regions)
    torch.testing.assert_close(train, dense, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(evaluation, train, atol=2e-6, rtol=2e-6)


def test_hash_projectors_receive_training_gradients():
    torch.manual_seed(7)
    a = HashAttention(12, 3, window_size=3, num_bits=8)
    x = torch.randn(2, 9, 12, requires_grad=True)
    y, aux = a(x, torch.ones(2, 9, dtype=torch.bool), torch.zeros(2, 9, dtype=torch.long))
    (y.square().mean() + aux).backward()
    for p in [a.query_projector.weight, a.key_projector.weight, a.prototypes.latents]:
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


def test_compensation_reuses_but_does_not_update_key_projector():
    torch.manual_seed(2)
    a = HashAttention(8, 2, window_size=2, num_bits=4, radius=0)
    with torch.no_grad():
        a.key_projector.weight.zero_()
        a.prototypes.latents.fill_(-1.)
    miner = HashMiner(8, 2)
    x = torch.randn(1, 4, 8, requires_grad=True)
    valid = torch.tensor([[True, True, False, True]])
    y, selected = miner(x, valid, torch.zeros(1, 4, dtype=torch.long), a, torch.nn.LayerNorm(8))
    assert torch.equal(selected, valid)
    torch.testing.assert_close(y[:, 2], x[:, 2])
    y.square().sum().backward()
    assert a.key_projector.weight.grad is None
    assert a.qkv.weight.grad is None
    assert miner.attn.in_proj_weight.grad is not None


def test_union_foreground_ratio_clips_and_does_not_double_count():
    assert foreground_ratio(torch.tensor([[0., 0., 10., 10.], [5., 5., 15., 15.]]), (20, 20)) == pytest.approx(175 / 400)
    assert foreground_ratio(torch.empty(0, 4), (20, 20)) == 0
    assert foreground_ratio(torch.tensor([[-5., -5., 30., 30.]]), (20, 20)) == 1


def test_odd_size_backbone_forward_backward_and_adaptive_selection():
    torch.manual_seed(8)
    torch.set_num_threads(2)
    model = BiHashFormer(embed_dims=12, depths=(1, 1, 1, 1),
                        num_heads=(3, 3, 6, 12), window_size=3,
                        num_bits=8, radius=1, roi_pool_sizes=(2, 2, 1, 1),
                        roi_hidden=8, drop_path_rate=0., with_cp=True)
    model.train()
    x = torch.randn(2, 3, 65, 71)
    outputs = model(x)
    assert [list(t.shape) for t in outputs] == [[2, 12, 17, 18], [2, 24, 9, 9], [2, 48, 5, 5], [2, 96, 3, 3]]
    losses = model.auxiliary_losses(torch.tensor([.1, .8]))
    total = sum(t.square().mean() for t in outputs) + sum(losses.values())
    total.backward()
    assert torch.isfinite(total)
    assert model.stages[0].ratio_predictor[0].weight.grad.abs().sum() > 0
    assert model.stages[0].blocks[0].score_net.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        assert all(torch.isfinite(t).all() for t in model(x))
