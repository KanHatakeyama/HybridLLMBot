import openai
import numpy as np
import time


class GPTEmbedding:
    def __init__(self, api_key, model="text-embedding-ada-002",
                 sleep_time=0.9) -> None:
        self.model = model
        openai.api_key = api_key
        self.sleep_time = sleep_time

    def __call__(self, text):
        v = openai.Embedding.create(input=[text], model=self.model)[
            'data'][0]['embedding']

        # this may be necessary to avoid the error of "Too Many Requests"
        time.sleep(self.sleep_time)

        return np.array(v).reshape(1, -1).astype('float32')
