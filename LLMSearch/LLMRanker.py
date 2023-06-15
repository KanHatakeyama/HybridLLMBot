
import ctranslate2
import transformers
import torch
import numpy as np

class LLMRanker:
    def __init__(self) -> None:

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = ctranslate2.Generator("./rinna_ppo", device=device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "rinna/japanese-gpt-neox-3.6b-instruction-ppo", use_fast=False)
        self.temperature=0.71

    def query(self,query):

        prompt =f"ユーザー:{query}<NL>システム: "
        # 推論の実行
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(prompt, add_special_tokens=False)
        )
        results = self.generator.generate_batch(
            [tokens],
            max_length=64,
            sampling_topk=10,
            sampling_temperature=self.temperature,
            include_prompt_in_result=False,
        )
        text = self.tokenizer.decode(results[0].sequences_ids[0])
        return text

    def compare_text(self,t1,t2,evaluation=True,verbose=False):
        query=f"""
        指示:｢{t1}｣という質問は｢{t2}｣と関係ある?
        出力:Yes/Noを出力
        """
        text=self.query(query)
        if verbose:
            print(text)
        if evaluation:
            eval=self.eval_pos_neg(text)
            if verbose:
                print(eval)
            return eval
        return text
    
    def eval_pos_neg(self,ans):
        eval=0
        p_position=np.inf
        n_position=np.inf
        ans=ans.replace("Yes/No","")

        for pos_text in["はい","yes","Yes","YES","関係あります"]:
            c_p_position=ans.find(pos_text)
            if c_p_position!=-1:
                eval+=1
                p_position=min(p_position,c_p_position)
        for neg_text in["いいえ","no","No","NO","関係ありません"]:
            c_n_position=ans.find(neg_text)
            if c_n_position!=-1:
                eval-=1
                n_position=min(n_position,c_n_position)

        if n_position<np.inf and p_position<np.inf:
            if n_position<p_position:
                return -1
            else:
                return 1

        if eval==0:
            return 0
        eval=eval/abs(eval)
        return eval
    
    def eval_text(self,t1,t2,n=10,threshold=0,verbose=False,return_list=True):
        eval=0
        res_list=[]
        for i in range(n):
            res=self.compare_text(t1,t2,evaluation=True,verbose=verbose)
            res_list.append(res)
            eval+=res

        if return_list:
            return res_list


        eval=eval/n
        return eval
        if eval>threshold:
            return True
        return False