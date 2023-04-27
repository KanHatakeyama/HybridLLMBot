from rwkv.model import RWKV
from rwkv.utils import PIPELINE
import numpy as np

from langchain.llms.base import LLM

class RWKVEmbedding:
    def __init__(self,settings) -> None:

        self.model = RWKV(model=settings["RWKV_MODEL"], 
                    strategy=settings["RWKV_STRATEGY"],
                    )
        self.pipeline = PIPELINE(self.model,settings["RWKV_TOKENIZER"])
        print("RWKV initiated successfully")

    def __call__(self,txt,mean_mode=True):
        inp_vec=self.pipeline.encode(txt)
        out, state = self.model.forward(inp_vec, None)
        np_state=[i.detach().cpu().numpy() for i in state]
        if mean_mode:
            np_state=np.mean(np_state,axis=0)
        else:
            np_state=np.concatenate(np_state, axis=0)

        return np_state.reshape(1,-1)
