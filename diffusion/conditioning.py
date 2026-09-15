"""Section/domain embeddings and classifier-free condition dropout."""

from __future__ import annotations

import torch
from torch import nn

SOURCE_DOMAIN_ID = 0
TARGET_DOMAIN_ID = 1
UNKNOWN_DOMAIN_ID = 2

DOMAIN_TO_ID = {
    "source": SOURCE_DOMAIN_ID,
    "target": TARGET_DOMAIN_ID,
    "unknown": UNKNOWN_DOMAIN_ID,
}


class MetadataConditioner(nn.Module):
    """Project section and domain labels into the U-Net embedding space."""

    def __init__(self, config: dict, embedding_dim: int):
        super().__init__()
        conditioning_config = config["conditioning"]
        self.use_section = bool(conditioning_config.get("use_section", False))
        self.use_domain = bool(conditioning_config.get("use_domain", False))
        self.dropout = float(conditioning_config.get("condition_dropout", 0.0))
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("condition_dropout must satisfy 0 <= p < 1")
        self.num_sections = int(conditioning_config.get("num_sections", 3))
        if self.use_section and self.num_sections < 1:
            raise ValueError("num_sections must be positive when section conditioning is used")
        self.unknown_section_id = self.num_sections

        input_parts = 0
        if self.use_section:
            self.section_embedding = nn.Embedding(self.num_sections + 1, embedding_dim)
            input_parts += 1
        else:
            self.section_embedding = None
        if self.use_domain:
            self.domain_embedding = nn.Embedding(len(DOMAIN_TO_ID), embedding_dim)
            input_parts += 1
        else:
            self.domain_embedding = None
        if input_parts == 0:
            raise ValueError("MetadataConditioner requires at least one enabled condition")

        self.projection = nn.Sequential(
            nn.Linear(input_parts * embedding_dim, embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    @staticmethod
    def _prepare_ids(
        ids: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        unknown_id: int,
        upper_bound: int,
        name: str,
    ) -> torch.Tensor:
        if ids is None:
            return torch.full(
                (batch_size,), unknown_id, device=device, dtype=torch.long
            )
        ids = ids.to(device=device, dtype=torch.long)
        if ids.ndim == 0:
            ids = ids.expand(batch_size)
        if ids.shape != (batch_size,):
            raise ValueError(f"One {name} id is required per batch element")
        if bool(((ids < 0) | (ids >= upper_bound)).any()):
            raise ValueError(f"{name} ids must be in [0, {upper_bound - 1}]")
        return ids

    def forward(
        self,
        section_id: torch.Tensor | None,
        domain_id: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        section_ids = None
        domain_ids = None
        if self.use_section:
            section_ids = self._prepare_ids(
                section_id,
                batch_size,
                device,
                self.unknown_section_id,
                self.num_sections + 1,
                "section",
            )
        if self.use_domain:
            domain_ids = self._prepare_ids(
                domain_id,
                batch_size,
                device,
                UNKNOWN_DOMAIN_ID,
                len(DOMAIN_TO_ID),
                "domain",
            )

        if self.training and self.dropout > 0.0:
            drop_mask = torch.rand(batch_size, device=device) < self.dropout
            if section_ids is not None:
                section_ids = torch.where(
                    drop_mask,
                    torch.full_like(section_ids, self.unknown_section_id),
                    section_ids,
                )
            if domain_ids is not None:
                domain_ids = torch.where(
                    drop_mask,
                    torch.full_like(domain_ids, UNKNOWN_DOMAIN_ID),
                    domain_ids,
                )

        parts: list[torch.Tensor] = []
        if self.section_embedding is not None:
            parts.append(self.section_embedding(section_ids))
        if self.domain_embedding is not None:
            parts.append(self.domain_embedding(domain_ids))
        return self.projection(torch.cat(parts, dim=1))
