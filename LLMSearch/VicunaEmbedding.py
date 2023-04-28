from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch

def calc_vec(input_text, model, tokenizer):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    #list形式でtensorが入っているoutputsを1次元のnumpyに変換
    vec_list=[]
    for v in outputs[-1]:
        mean_v=v.mean(axis=1)
        vec_list.append(mean_v.numpy().flatten())

    vec=np.array(vec_list).flatten()

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
            norm = np.linalg.norm(np_state, axis=0, keepdims=True)
            np_state= np_state/ norm

        return np_state.reshape(1,-1).astype('float32')