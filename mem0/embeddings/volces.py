import os
from typing import Optional

from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

import numpy as np

class VolcesEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "ep-20241023114911-txvff"
        self.config.embedding_dims = self.config.embedding_dims or 512

        api_key = self.config.api_key or os.getenv("ARK_API_KEY")
        base_url = self.config.volces_base_url or os.getenv("VOLCES_API_BASE")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _sliced_norm_l2(self, vec: list[float], dim=512) -> list[float]:
        norm = float(np.linalg.norm(vec[:dim]))
        return [v / norm for v in vec[:dim]]

    def embed(self, text):
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        origin_vector = self.client.embeddings.create(input=[text], model=self.config.model).data[0].embedding
        if len(origin_vector) > self.config.embedding_dims:
            final_vec = self._sliced_norm_l2(origin_vector, self.config.embedding_dims)
        else:
            final_vec = origin_vector
        return final_vec
