from rwkv.model import RWKV
from rwkv.utils import PIPELINE
import numpy as np


class RWKVEmbedding:
    def __init__(self,model_path,
                tokenizer_path="settings/20B_tokenizer.json",
                 strategy='cpu fp32',
                 mean_mode=True,normalize=True) -> None:

        self.model = RWKV(model=model_path, 
                    strategy=strategy,
                    )
        self.pipeline = PIPELINE(self.model,tokenizer_path)

        self.mean_mode=mean_mode
        self.normalize=normalize

        print("RWKV initiated successfully")

    def __call__(self,txt):
        inp_vec=self.pipeline.encode(txt)
        out, state = self.model.forward(inp_vec, None)
        np_state=[i.detach().cpu().numpy() for i in state]

        np_state=[i for i in np_state if not np.isnan(i).any()]
        np_state=[i for i in np_state if np.linalg.norm(i, axis=0, keepdims=True)<10**2]
        if self.mean_mode:
            np_state=np.mean(np_state,axis=0)
        else:
            np_state=np.concatenate(np_state, axis=0)

        if self.normalize:
            norm = np.linalg.norm(np_state, axis=0, keepdims=True)
            np_state= np_state/ norm

        return np_state.reshape(1,-1).astype('float32')
