"""DINOv3 frozen vision backbone for feature extraction."""

import torch
from transformers import AutoModel


class DINOv3Backbone:
    """Wraps the DINOv3 ViT model as a frozen feature extractor.

    Extracts the CLS token embedding (768-dim for ViT-B/16) from images.
    All backbone weights are frozen -- only downstream heads should train.
    """

    def __init__(self, model_name: str, device: str = "cpu", hf_token: str = None):
        self.device = device
        print(f"Loading DINOv3 backbone: {model_name}")
        self.model = AutoModel.from_pretrained(
            model_name,
            token=hf_token,
        )
        self.model.to(device)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        print(f"DINOv3 loaded on {device} (frozen, {self._count_params()}M params)")

    def _count_params(self) -> str:
        total = sum(p.numel() for p in self.model.parameters())
        return f"{total / 1e6:.1f}"

    @torch.no_grad()
    def extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract CLS token embeddings.

        Args:
            pixel_values: (batch, 3, 224, 224) normalized image tensors.

        Returns:
            (batch, embedding_dim) CLS token features.
        """
        pixel_values = pixel_values.to(self.device)
        outputs = self.model(pixel_values=pixel_values)
        # CLS token is the first token in the sequence
        return outputs.last_hidden_state[:, 0]

    @staticmethod
    def download(model_name: str, hf_token: str = None):
        """Pre-download model weights to HuggingFace cache."""
        print(f"Downloading {model_name}...")
        AutoModel.from_pretrained(model_name, token=hf_token)
        print("Download complete. Model cached locally.")
