import joblib
class DictEmbedder:
    def __init__(self,dict_path,embedder):
        try:
            self.dict=joblib.load(dict_path)
        except:
            print("prepare new dict")
            self.dict={}
        self.embedder=embedder
        self.dict_path=dict_path

    def __call__(self,text):
        if text in self.dict:
            return self.dict[text].reshape(-1)

        #calc
        vec=self.embedder(text)
        self.dict[text]=vec
        joblib.dump(self.dict,self.dict_path)


        return vec.reshape(-1)