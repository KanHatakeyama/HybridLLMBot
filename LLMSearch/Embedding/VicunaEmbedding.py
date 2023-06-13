from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch

def calc_vec(input_text, model, tokenizer):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    vec_list=[]
    for v in outputs[-1]:
        vec_list.append(v.numpy())

    vec=np.array(vec_list)
    print(vec.shape)
    vec=np.mean(vec,axis=2)
    vec=np.mean(vec,axis=0)
    print(vec.shape)

    return vec

class VicunaEmbedding:
    def __init__(self,model_name="AlekseyKorshuk/vicuna-7b",normalize=True) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = model
        self.tokenizer = tokenizer
        self.normalize=normalize

        print("Vicuna initiated successfully")

    def __call__(self,txt):
        np_state=calc_vec(txt,self.model,self.tokenizer)

        if self.normalize:
            norm = np.linalg.norm(np_state, axis=1, keepdims=True)
            np_state= np_state/ norm

        return np_state.reshape(1,-1).astype('float32')