from types import SimpleNamespace

import torch

from openpi.models_pytorch.preprocessing_pytorch import IMAGE_KEYS, preprocess_observation_pytorch
from openpi.training.router_labels import positive_distribution_loss


def test_sparse_positive_loss_skips_unlabeled_and_invalid_positions():
    scores = torch.tensor([[0., 1., 3.], [5., -3., 20.]], requires_grad=True)
    targets = torch.tensor([[1., 2., 9.], [0., 0., 0.]])
    valid = torch.tensor([[True, True, False], [True, True, False]])
    loss, count = positive_distribution_loss(scores, targets, valid)
    reference = -(torch.tensor([1/3, 2/3])*scores[0,:2].log_softmax(0)).sum()
    torch.testing.assert_close(loss, reference)
    assert count == 1
    loss.backward()
    assert torch.equal(scores.grad[1], torch.zeros(3))
    assert scores.grad[0,2] == 0
    scores.grad = None
    empty, count = positive_distribution_loss(scores, torch.zeros_like(targets), valid)
    empty.backward()
    assert empty == count == 0 and torch.equal(scores.grad, torch.zeros_like(scores))


def test_label_geometry_tracks_crop_rotation_but_not_color(monkeypatch):
    torch.manual_seed(42)
    mask = torch.zeros(2,224,224)
    mask[:,50:100,80:145] = 1
    images = {key: (mask[:,None].expand(-1,3,-1,-1)*2-1).clone() for key in IMAGE_KEYS}
    obs = SimpleNamespace(
        images=images, image_masks={k: torch.ones(2,dtype=torch.bool) for k in IMAGE_KEYS},
        router_label_masks={k:mask.clone() for k in IMAGE_KEYS}, state=torch.zeros(2,8),
        tokenized_prompt=None, tokenized_prompt_mask=None, token_ar_mask=None, token_loss_mask=None,
    )
    calls = 0

    def random_value(*shape, **kwargs):
        nonlocal calls
        calls += 1
        # Rotate the external view by 3 degrees; keep all color factors at exactly one.
        return torch.full(shape, .8 if calls == 1 else .5, **kwargs)

    monkeypatch.setattr(torch, 'rand', random_value)
    processed = preprocess_observation_pytorch(obs, train=True)
    for key in IMAGE_KEYS:
        torch.testing.assert_close((processed.images[key][:,0]+1)/2, processed.router_label_masks[key], atol=1e-6, rtol=0)
    assert not torch.equal(mask, processed.router_label_masks['base_0_rgb'])
    assert torch.equal(mask, processed.router_label_masks['left_wrist_0_rgb'])
