import faiss
import joblib
import os
from .DocSplitter import split_text
import json

class VectorSearcher:
    def __init__(self, embedder, 
                 setting_path='settings/settings.json',
                 chunk_size_limit=None,
) -> None:
        with open(setting_path) as f:
            settings = json.load(f)
        self.embedder = embedder
        self.base_path =settings["data_path"] 
        if chunk_size_limit is None:
            self.chunk_size_limit = settings["chunk_size_limit"]
        else:
            self.chunk_size_limit = chunk_size_limit
        self.path_list = []
        self.dim = self.embedder("a").shape[1]
        #self.index = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexFlatIP(self.dim)

        meta_path = self.base_path+"/meta"
        if not os.path.exists(meta_path):
            os.mkdir(meta_path)

        try:
            self.load_model()
        except:
            print("error loading model. initializing...")

    def initialize(self):
        self.path_list = []
        self.index = faiss.IndexFlatIP(self.dim)

    def add_text(self, path):
        if path in self.path_list:
            return

        with open(path, 'r') as f:
            text = f.read()
        chunk_list = split_text(text, self.chunk_size_limit)

        for _, chunk in enumerate(chunk_list):
            self.add_record(chunk,path)

        self.save_model()

    def add_record(self, text,path=""):
        vec = self.embedder(text)
        self.index.add(vec)
        self.path_list.append(path)
        return vec

    def save_model(self):
        faiss.write_index(self.index, self.base_path+"/meta/faiss.index")
        joblib.dump(self.path_list, self.base_path+"/meta/path_list.bin")

    def load_model(self):
        self.index = faiss.read_index(self.base_path+"/meta/faiss.index")
        self.path_list = joblib.load(self.base_path+"/meta/path_list.bin")

    def search(self, txt, k=3):

        vec = self.embedder(txt)
        D, I = self.index.search(vec, k)

        ret_list = []
        for i in range(k):
            path = self.path_list[I[0][i]]
            with open(path, 'r', encoding="utf-8") as f:
                text = f.read()

            temp_dict = {}
            temp_dict["path"] = path
            temp_dict["text"] = text
            temp_dict["sim"] = D[0][i]
            ret_list.append(temp_dict)

        return ret_list
