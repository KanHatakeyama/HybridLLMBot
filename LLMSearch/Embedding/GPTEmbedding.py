import openai
import numpy as np

class GPTEmbedding:
    def __init__(self,api_key,model="text-embedding-ada-002") -> None:
        self.model=model
        openai.api_key =  api_key
    def __call__(self,text):
        v=openai.Embedding.create(input = [text], model=self.model)['data'][0]['embedding']

        return np.array(v).reshape(1,-1).astype('float32')

