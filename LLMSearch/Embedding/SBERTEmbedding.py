from sentence_transformers import SentenceTransformer
import torch


class SBERTEmbedding:
    def __init__(self, model_name="all-mpnet-base-v2", device=None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, text):
        v = self.model.encode([text])[0]
        return v.reshape(1, -1).astype('float32')
