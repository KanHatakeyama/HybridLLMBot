import joblib
class DictEmbedder:
    def __init__(self,dict_path,embedder,init=False):
        if init:
            self.dict={}
        else:
            try:
                self.dict=joblib.load(dict_path)
            except:
                print("prepare new dict")
                self.dict={}
        self.embedder=embedder
        self.dict_path=dict_path
        self.count=0

    def __call__(self,text):
        if text in self.dict:
            return self.dict[text].reshape(-1)

        #calc
        vec=self.embedder(text)
        self.dict[text]=vec



        self.count+=1
        if self.count%10==0:
            joblib.dump(self.dict,self.dict_path)
        if self.count%100==0:
            joblib.dump(self.dict,self.dict_path+".backup")

        return vec.reshape(-1)

    def save(self):
        joblib.dump(self.dict,self.dict_path)