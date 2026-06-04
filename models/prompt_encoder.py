# =============================================================================
# models/prompt_encoder.py — Prompt Encoder
#
# Encodes three types of prompts into sparse + dense embeddings:
#
#   1. BOX PROMPTS (geometric): Bounding boxes [x1, y1, x2, y2]
#      - Encoded as positional embeddings of two corner points
#      - Same approach as original SAM
#
#   2. MODALITY PROMPTS (from MCP-MedSAM): Learnable embedding per modality
#      - Each modality (CT, MRI, PET, etc.) gets a unique learnable vector
#      - Conditions the model on imaging physics
#
#   3. CONTENT PROMPTS (from MCP-MedSAM): Box-crop CNN features
#      - Extract a small feature from inside the bounding box
#      - Provides content context: "what does the target look like?"
#
# The encoder outputs:
#   - sparse_embeddings: (B, N_prompts, embed_dim) — for attention in decoder
#   - dense_embeddings:  (B, embed_dim, H, W)      — added to image features
#
# Reference: MCP-MedSAM (CVPR 2024 challenge winner)
#            SAM (Kirillov et al., 2023)
# =============================================================================

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbeddingRandom(nn.Module):
    """
    Random Fourier Feature positional encoding.
    Maps 2D normalised coordinates (x, y) in [0, 1] to a high-dim embedding.

    This is the same encoding used in the original SAM paper.
    It allows the model to represent arbitrary box coordinates continuously.

    Args:
        num_pos_feats (int): Half the output dimension. Output is 2 * num_pos_feats.
        scale (float): Gaussian scale for random projection matrix.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        # Gaussian random matrix — fixed (not learned), registered as buffer
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode normalised coords in [0, 1].

        Args:
            coords: (..., 2) normalised coordinates

        Returns:
            pe: (..., 2 * num_pos_feats)
        """
        # Map to [-1, 1]
        coords = 2 * coords - 1

        # Project: (..., 2) @ (2, num_pos_feats) -> (..., num_pos_feats)
        coords = coords @ self.positional_encoding_gaussian_matrix.to(coords.dtype)
        coords = 2 * torch.pi * coords

        # Fourier features: sin and cos concatenated
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate a dense position embedding for a grid of (H, W) positions.

        Args:
            size: (H, W) — spatial size of the feature map

        Returns:
            pe: (1, 2*num_pos_feats, H, W)
        """
        H, W = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((H, W), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5   # Y coords: 0.5, 1.5, ..., H-0.5
        x_embed = grid.cumsum(dim=1) - 0.5   # X coords: 0.5, 1.5, ..., W-0.5
        y_embed = y_embed / H
        x_embed = x_embed / W
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))  # (H, W, 2F)
        return pe.permute(2, 0, 1).unsqueeze(0)  # (1, 2F, H, W)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Encode a set of (x, y) point coordinates.

        Args:
            coords_input: (B, N, 2) — pixel coordinates in [0, img_size]
            image_size: (H, W) — used to normalise coords to [0, 1]

        Returns:
            pe: (B, N, 2 * num_pos_feats)
        """
        coords = coords_input.clone().float()
        coords[..., 0] /= image_size[1]  # Normalise x by W
        coords[..., 1] /= image_size[0]  # Normalise y by H
        return self._pe_encoding(coords)


class ContentPromptEncoder(nn.Module):
    """
    Content Prompt from MCP-MedSAM.

    Extracts a compact feature vector from the region inside the bounding box.
    This tells the model "what the target looks like", helping disambiguate
    between similar-looking structures (e.g., liver vs spleen).

    Design:
      - RoI-pool the image features at the box location
      - Pass through a lightweight CNN
      - Output a single embedding vector per box

    Args:
        embed_dim (int): Output embedding dimension.
        content_dim (int): Internal CNN dimension.
        roi_size (int): Size of RoI-pooled feature (content_dim x roi_size x roi_size).
    """

    def __init__(self, embed_dim: int, content_dim: int = 64, roi_size: int = 7):
        super().__init__()

        self.roi_size = roi_size

        # Small CNN to process the RoI crop
        self.cnn = nn.Sequential(
            nn.Conv2d(embed_dim, content_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(content_dim),
            nn.GELU(),
            nn.Conv2d(content_dim, content_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(content_dim),
            nn.GELU(),
        )

        # Global average pool + linear to produce final embedding
        self.pool_and_project = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, content_dim, 1, 1)
            nn.Flatten(),                  # (B, content_dim)
            nn.Linear(content_dim, embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        image_features: torch.Tensor,   # (B, embed_dim, H, W)
        boxes_norm: torch.Tensor,        # (B, 4) normalised boxes [x1,y1,x2,y2] in [0,1]
    ) -> torch.Tensor:
        """
        Extract content-aware features from inside each bounding box.

        Args:
            image_features: (B, embed_dim, H, W)
            boxes_norm: (B, 4) in [0, 1]

        Returns:
            content_embedding: (B, embed_dim)
        """
        B, C, H, W = image_features.shape

        # Crop and resize each RoI using grid_sample
        # Create a grid for each box
        content_embeds = []
        for i in range(B):
            x1, y1, x2, y2 = boxes_norm[i]  # normalised coordinates

            # Build a sampling grid for this RoI
            # grid_sample expects coordinates in [-1, 1]
            xs = torch.linspace(
                2 * x1 - 1, 2 * x2 - 1, self.roi_size, device=image_features.device
            )
            ys = torch.linspace(
                2 * y1 - 1, 2 * y2 - 1, self.roi_size, device=image_features.device
            )
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, rs, rs, 2)

            # Sample from feature map
            roi = F.grid_sample(
                image_features[i:i+1],  # (1, C, H, W)
                grid,
                mode="bilinear",
                align_corners=False,
            )  # (1, C, roi_size, roi_size)

            content_embeds.append(roi)

        rois = torch.cat(content_embeds, dim=0)  # (B, C, roi_size, roi_size)

        # Process with CNN
        rois = self.cnn(rois)  # (B, content_dim, roi_size, roi_size)

        # Pool to get a single vector per sample
        return self.pool_and_project(rois)  # (B, embed_dim)


class PromptEncoder(nn.Module):
    """
    Complete prompt encoder combining:
      - Box positional embeddings (sparse)
      - Modality learnable embeddings (sparse)
      - Content prompt from box crop (sparse)
      - Dense positional embedding for image features

    Args:
        embed_dim (int): Embedding dimension for all prompts.
        image_size (int): Input image size (used for coordinate normalisation).
        image_embedding_size (int): Feature map size (= image_size // 16).
        num_modalities (int): Number of imaging modalities.
        modality_embed_dim (int): Dimension of learnable modality embedding.
        content_embed_dim (int): Dimension of content prompt CNN.
    """

    def __init__(
        self,
        embed_dim: int,
        image_size: int = 512,
        image_embedding_size: int = 32,  # 512 / 16 = 32
        num_modalities: int = 11,
        modality_embed_dim: int = 64,
        content_embed_dim: int = 64,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.image_size = image_size
        self.image_embedding_size = image_embedding_size

        # ----- Positional encoding for box corners -----
        # We use 64 Fourier features → 128-dim encoding per point
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Special tokens for box corners (top-left and bottom-right)
        # These distinguish "which corner" the position belongs to
        self.corner_tl_embed = nn.Embedding(1, embed_dim)  # Top-left corner
        self.corner_br_embed = nn.Embedding(1, embed_dim)  # Bottom-right corner

        # Padding point token (for when no box is provided)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # ----- Modality prompt (learnable embedding per modality) -----
        # Modality indices: 0=CT, 1=MRI, 2=PET, 3=X-ray, 4=US, 5=Mammo,
        #                   6=OCT, 7=Endo, 8=Fundus, 9=Derm, 10=Microscopy
        self.modality_embed = nn.Embedding(num_modalities, embed_dim)

        # ----- Content prompt -----
        self.content_encoder = ContentPromptEncoder(
            embed_dim=embed_dim,
            content_dim=content_embed_dim,
        )

        # ----- Dense positional embedding -----
        # Added to image features to inject spatial position information
        self.dense_pe = PositionEmbeddingRandom(embed_dim // 2)

    def _embed_boxes(
        self,
        boxes: torch.Tensor,   # (B, 4) in pixel coords [x1, y1, x2, y2]
    ) -> torch.Tensor:
        """
        Encode bounding box as two corner point embeddings.

        Returns:
            box_embeddings: (B, 2, embed_dim) — one embed per corner
        """
        # Shift by 0.5 to get centre of each pixel
        boxes = boxes + 0.5

        # Reshape to (B, 2, 2): [[x1,y1], [x2,y2]]
        coords = boxes.reshape(-1, 2, 2)

        # Get positional encodings for the two corners
        corner_embeds = self.pe_layer.forward_with_coords(
            coords, (self.image_size, self.image_size)
        )  # (B, 2, embed_dim)

        # Add learnable corner tokens to distinguish TL from BR
        corner_embeds[:, 0, :] += self.corner_tl_embed.weight  # Top-left
        corner_embeds[:, 1, :] += self.corner_br_embed.weight  # Bottom-right

        return corner_embeds  # (B, 2, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the dense positional embedding for the image feature map.

        Shape: (1, embed_dim, H_feat, W_feat)
        """
        return self.dense_pe(
            (self.image_embedding_size, self.image_embedding_size)
        )

    def forward(
        self,
        boxes: torch.Tensor,            # (B, 4) pixel coords [x1, y1, x2, y2]
        modality_ids: torch.Tensor,     # (B,) integer modality index
        image_features: torch.Tensor,  # (B, embed_dim, H_feat, W_feat)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all prompts.

        Args:
            boxes: (B, 4) bounding boxes in pixel space
            modality_ids: (B,) integer in [0, num_modalities)
            image_features: (B, embed_dim, H_feat, W_feat) from encoder

        Returns:
            sparse_embeddings: (B, 4, embed_dim)
                [box_tl, box_br, modality, content]
            dense_embeddings:  (B, embed_dim, H_feat, W_feat)
                Dense positional embedding (same for all samples in batch)
        """
        B = boxes.shape[0]
        device = boxes.device

        # ---- (1) Box prompt: two corner embeddings ----
        box_embeds = self._embed_boxes(boxes)  # (B, 2, embed_dim)

        # ---- (2) Modality prompt ----
        mod_embeds = self.modality_embed(modality_ids)  # (B, embed_dim)
        mod_embeds = mod_embeds.unsqueeze(1)             # (B, 1, embed_dim)

        # ---- (3) Content prompt ----
        # Normalise boxes to [0, 1] for the content encoder
        boxes_norm = boxes.float().clone()
        boxes_norm[:, [0, 2]] /= self.image_size  # normalise x by width
        boxes_norm[:, [1, 3]] /= self.image_size  # normalise y by height
        boxes_norm = boxes_norm.clamp(0, 1)

        content_embeds = self.content_encoder(image_features, boxes_norm)  # (B, embed_dim)
        content_embeds = content_embeds.unsqueeze(1)  # (B, 1, embed_dim)

        # ---- Concatenate all sparse prompts ----
        # Total: 2 (box corners) + 1 (modality) + 1 (content) = 4 tokens
        sparse_embeddings = torch.cat(
            [box_embeds, mod_embeds, content_embeds], dim=1
        )  # (B, 4, embed_dim)

        # ---- Dense positional embedding ----
        # Generate PE matching the ACTUAL feature map size (not the configured one,
        # which may differ if input resolution != model.img_size).
        _, _, Hf, Wf = image_features.shape
        if Hf == self.image_embedding_size and Wf == self.image_embedding_size:
            dense_embeddings = self.get_dense_pe().to(device).expand(B, -1, -1, -1)
        else:
            dense_embeddings = self.dense_pe((Hf, Wf)).to(device).expand(B, -1, -1, -1)
        # (B, embed_dim, H_feat, W_feat)

        return sparse_embeddings, dense_embeddings
