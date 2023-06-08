
from sentence_transformers.cross_encoder import CrossEncoder
import torch
import json


# TODO: somehow, cross encoder init in the model does not work
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device=device)


class CrossEncoder:
    def __init__(self,
                 setting_path='settings/settings.json',
                 device="cpu"):

        with open(setting_path) as f:
            settings = json.load(f)
        #model_name = settings["CROSS_ENCODER_MODEL"]
        #self.model = CrossEncoder(model_name, device=device)

        #self.model = model

    def predict(self, question, references):
        pairs = [(question, reference["text"]) for reference in references]

        scores = model.predict(pairs)
        for reference, score in zip(references, scores):
            reference["sim"] = score

        return sorted(references, key=lambda x: x['sim'], reverse=True)
