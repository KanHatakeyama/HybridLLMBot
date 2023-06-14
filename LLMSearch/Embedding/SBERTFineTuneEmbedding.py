import torch
from sentence_transformers import SentenceTransformer, models

class SBERTFineTuneEmbedding:
    def __init__(self, model_name='sonoisa/sentence-bert-base-ja-mean-tokens-v2', device=None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        bert = models.Transformer(model_name)
        pooling = models.Pooling(bert.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[bert, pooling])

    def __call__(self, text):
        v = self.model.encode([text])[0]
        return v.reshape(1, -1).astype('float32')