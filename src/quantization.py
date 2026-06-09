"""One-shot KMeans quantization matching geo32.py."""

from dataclasses import dataclass

import torch
from tqdm import tqdm


@dataclass
class Codebook:
    centers: torch.Tensor
    indices: torch.Tensor
    value_shape: tuple


def nearest_indices(feat, centers, chunk_size=10000):
    """Assign every feature vector to its nearest center."""

    chunks = []
    for start in range(0, feat.shape[0], chunk_size):
        dist = torch.cdist(
            feat[start : start + chunk_size].unsqueeze(0),
            centers.unsqueeze(0),
        )[0]
        chunks.append(torch.argmin(dist, dim=-1))
    return torch.cat(chunks, dim=0)


def update_centers(feat, indices, centers):
    """Update centers for fixed assignments without padded cluster tables."""

    sums = torch.zeros_like(centers)
    counts = torch.zeros(centers.shape[0], dtype=feat.dtype, device=feat.device)
    sums.index_add_(0, indices, feat)
    counts.index_add_(0, indices, torch.ones_like(indices, dtype=feat.dtype))

    updated = torch.zeros_like(centers)
    nonempty = counts > 0
    updated[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(-1)
    return updated


def fit_codebook(
    values,
    num_clusters=4096,
    num_iters=10,
    chunk_size=10000,
    initial_centers=None,
):
    """Run KMeans and return assignments matching the final centers."""

    original_shape = values.shape
    feat = values.detach().reshape(values.shape[0], -1)
    num_clusters = max(1, min(num_clusters, feat.shape[0]))
    if initial_centers is None or initial_centers.shape != (num_clusters, feat.shape[1]):
        initial_ids = torch.randperm(feat.shape[0], device=feat.device)[:num_clusters]
        centers = feat[initial_ids].clone()
    else:
        centers = initial_centers.detach().to(feat).clone()

    for _ in tqdm(range(num_iters), desc="  KMeans iters", unit="iter", leave=False):
        indices = nearest_indices(feat, centers, chunk_size=chunk_size)
        centers = update_centers(feat, indices, centers)

    # The final center update changes the Voronoi cells. Reassign once so the
    # serialized indices actually correspond to the serialized centers.
    indices = nearest_indices(feat, centers, chunk_size=chunk_size)
    return Codebook(
        centers=centers,
        indices=indices.reshape(original_shape[0]),
        value_shape=tuple(original_shape[1:]),
    )


def quantize_appearance_attributes(attributes, codebook_sizes=None, num_iters=10):
    codebook_sizes = codebook_sizes or {}
    codebooks = {}
    for name, values in tqdm(list(attributes.items()), desc="Quantizing attributes", unit="attr"):
        codebooks[name] = fit_codebook(
            values,
            num_clusters=codebook_sizes.get(name, 4096),
            num_iters=num_iters,
        )
    return codebooks
