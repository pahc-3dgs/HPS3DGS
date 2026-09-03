"""Owner (template) bookkeeping over a shared HAC++ anchor set.

The shared HAC++ backend encodes **one** anchor set for the whole scene. Each
anchor belongs to exactly one owner (a canonical component template); the
owner id plus the row index inside that owner is what lets the decoder rebuild
every instance by re-posing the same canonical anchors.
"""

from __future__ import annotations

import numpy as np
import torch


class OwnerError(ValueError):
    pass


def build_owner_index(template_rows: dict[int, torch.Tensor], num_basis: int):
    """Assign an owner id and an in-owner row to every template anchor.

    Parameters
    ----------
    template_rows:
        ``{template_id: LongTensor of rows into the basis anchor array}``.
    num_basis:
        Total number of basis anchors (rows of the exported HAC++ init cloud).

    Returns
    -------
    owner_id: ``LongTensor[num_basis]`` (-1 for anchors that belong to no
        template, i.e. free/static anchors).
    row_in_owner: ``LongTensor[num_basis]`` (-1 for non-template anchors).
    """

    owner_id = torch.full((num_basis,), -1, dtype=torch.long)
    row_in_owner = torch.full((num_basis,), -1, dtype=torch.long)
    counts: dict[int, int] = {}
    for template_id in sorted(template_rows):
        rows = template_rows[template_id].long().reshape(-1)
        if rows.numel() == 0:
            raise OwnerError("template %d is empty" % template_id)
        if int(rows.min()) < 0 or int(rows.max()) >= num_basis:
            raise OwnerError("template %d rows out of range" % template_id)
        taken = owner_id[rows] != -1
        if bool(taken.any()):
            raise OwnerError("anchor rows shared by multiple templates")
        owner_id[rows] = template_id
        local = torch.arange(rows.numel())
        row_in_owner[rows] = local
        counts[template_id] = int(rows.numel())
    return owner_id, row_in_owner, counts


def owner_blocks_are_contiguous(owner_id: torch.Tensor, anchor_order: torch.Tensor | None = None):
    """Check every owner occupies one contiguous block of the anchor order.

    HAC++ Morton-sorts anchors before GPCC encoding, so the decoder recovers
    anchors in that order; owners must stay contiguous in it for the per-owner
    template lookup to be a slice instead of a gather.
    """

    order = torch.arange(owner_id.shape[0]) if anchor_order is None else anchor_order.long()
    ids = owner_id[order]
    seen: set[int] = set()
    current: int | None = None
    for value in ids.tolist():
        if value < 0:
            continue
        if value != current:
            if value in seen:
                return False
            seen.add(value)
            current = value
    return True


def save_owners(
    path,
    owner_id: torch.Tensor,
    row_in_owner: torch.Tensor,
    template_ids: list[int],
    ordering: str = "morton",
):
    """Persist the owner sidecar next to the HAC++ streams."""

    path = np.savez_compressed(
        path,
        owner_id=owner_id.to(torch.int32).cpu().numpy(),
        row_in_owner=row_in_owner.to(torch.int32).cpu().numpy(),
        template_ids=np.asarray(sorted(template_ids), dtype=np.int32),
        ordering=np.asarray(ordering),
    )
    return path


def load_owners(path):
    data = np.load(path, allow_pickle=False)
    owner_id = torch.from_numpy(data["owner_id"].astype(np.int64))
    row_in_owner = torch.from_numpy(data["row_in_owner"].astype(np.int64))
    template_ids = [int(item) for item in data["template_ids"].tolist()]
    ordering = str(data["ordering"]) if "ordering" in data else "morton"
    return owner_id, row_in_owner, template_ids, ordering


def gather_template_rows(owner_id: torch.Tensor, row_in_owner: torch.Tensor, template_id: int):
    """Rows of the canonical anchors belonging to one template, in order."""

    mask = owner_id == int(template_id)
    rows = torch.where(mask)[0]
    local = row_in_owner[mask]
    return rows[torch.argsort(local)]
