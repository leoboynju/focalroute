"""Sparse positive-only Router targets, keyed by the pinned dataset row index."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def load_positive_pixels(directory: Path, variant: str):
    with np.load(directory / 'masks.npz') as data:
        maps = data['maps'].astype(np.float32)
    with np.load(directory / 'labels.npz') as data:
        positive = data[variant + '_positive'][:512].reshape(2,16,16)
    if maps.shape != (4,2,224,224) or not np.isfinite(maps).all() or np.any(maps < 0):
        raise ValueError(f'Invalid Router label mask: {directory}')
    pixels = np.zeros((3,224,224),np.float32)
    weights = (1., .6, .8, .3 if variant.startswith('arm_') else 0.)
    for weight, role in zip(weights, maps, strict=True):
        pixels[:2] += weight * role / max(float(role.sum()), 1e-12)
    pixels[:2] *= np.repeat(np.repeat(positive,14,axis=1),14,axis=2)
    return pixels


class SparseRouterLabelsDataset:
    def __init__(self, dataset, manifest_path: str, variant: str):
        self.dataset = dataset
        self.root = Path(manifest_path).resolve().parent
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest['status'] != 'complete' or manifest['dataset_frames'] != len(dataset):
            raise ValueError('Router labels must be complete and match the full dataset length.')
        if manifest['dataset_revision'] != 'a4336d589d589045d1c56423ffdf3b88a0e19b1f':
            raise ValueError('Unsupported Router label dataset snapshot.')
        self.records = {r['dataset_index']: r for r in manifest['samples']}
        if len(self.records) != len(manifest['samples']) or any(i < 0 or i >= len(dataset) for i in self.records):
            raise ValueError('Duplicate or out-of-range Router label index.')
        self.variant = variant

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        pixels = np.zeros((3, 224, 224), np.float32)
        record = self.records.get(int(index))
        if record is not None:
            pixels = load_positive_pixels(self.root / record['directory'], self.variant)
        sample['router_label_masks'] = dict(zip(('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'), pixels, strict=True))
        return sample


def positive_distribution_loss(scores, targets, valid_positions):
    """Only confident positives contribute targets; no BCE background-negative labels."""
    targets = targets.float() * valid_positions
    mass = targets.sum(dim=-1, keepdim=True)
    distribution = targets / mass.clamp_min(1e-12)
    log_probability = scores.float().masked_fill(~valid_positions, -1e9).log_softmax(dim=-1)
    local_sum = -(distribution * log_probability).sum()
    stats = torch.stack([(mass.squeeze(-1) > 0).float().sum(), local_sum.detach()])
    world_size = 1
    if dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(stats)
    count = stats[0].clamp_min(1)
    gradient_loss = local_sum * world_size / count
    loss = gradient_loss + (stats[1] / count - gradient_loss.detach())
    return loss, stats[0]
