from rwkv.model import RWKV
from rwkv.utils import PIPELINE
import numpy as np


class RWKVEmbedding:
    def __init__(self,settings,mean_mode=True,normalize=True) -> None:

        self.model = RWKV(model=settings["RWKV_MODEL"], 
                    strategy=settings["RWKV_STRATEGY"],
                    )
        self.pipeline = PIPELINE(self.model,settings["RWKV_TOKENIZER"])

        self.mean_mode=mean_mode
        self.normalize=normalize

        print("RWKV initiated successfully")

    def __call__(self,txt):
        inp_vec=self.pipeline.encode(txt)
        out, state = self.model.forward(inp_vec, None)
        np_state=[i.detach().cpu().numpy() for i in state]

        if self.mean_mode:
            np_state=np.mean(np_state,axis=0)
        else:
            np_state=np.concatenate(np_state, axis=0)

        if self.normalize:
            norm = np.linalg.norm(np_state, axis=0, keepdims=True)
            np_state= np_state/ norm

        return np_state.reshape(1,-1).astype('float32')
