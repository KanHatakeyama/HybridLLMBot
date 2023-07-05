from . DocSplitter import clean_text, split_text
import json
import MeCab
from .Embedding.BM25Transformer import BM25Transformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


mecab = MeCab.Tagger("-Owakati")



class BM25VecDB:
    def __init__(self,
                 setting_path='settings/settings.json',
                 parser=mecab,
                 ) -> None:

        with open(setting_path) as f:
            settings = json.load(f)

        self.db_path = settings["data_path"]+"/meta/text_bm25vec.db"
        self.model_path=self.db_path.replace(".db", ".vec")
        self.chunk_size_limit = settings["chunk_size_limit"]
        self.parser= parser
        self.vectorizer= Pipeline(steps=[
            ("CountVectorizer", CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')),
            ("BM25Transformer", BM25Transformer())
        ])
        self.vec=None

        try:
            self.db=joblib.load(self.db_path)
        except:
            self.initialize()
        try:
            self.vec,self.vectorizer=joblib.load(self.model_path)
        except:
            pass

    def initialize(self):
        self.db = {
            "fin_path": [],
            "text": [],
            #"vec": [],
            "split_id": [],
        }
        print("init database")


    def parse_text(self, text):
        return self.parser.parse(text).strip()

    def add_record(self, path, text, split_id):
        doc_wakati = self.parse_text(text)
        self.db["fin_path"].append(path)
        self.db["text"].append(doc_wakati)
        self.db["split_id"].append(split_id)


    def add_text(self, path):
        if path in self.db["fin_path"]:
            #print(f"{path} already added")
            return

        with open(path, 'r') as f:
            text = f.read()

        text = clean_text(text)

        chunk_list = split_text(text, self.chunk_size_limit)
        for i, chunk in enumerate(chunk_list):
            self.add_record(path, chunk, i)
            #print(f"{i} {path}")

    def save(self):
        joblib.dump(self.db, self.db_path)
        self.fit_model()

    def fit_model(self):
        self.vec=self.vectorizer.fit_transform(self.db["text"])
        joblib.dump([self.vec,self.vectorizer], self.model_path)

    def search(self, query, k=10):
        if self.vec is None:
            print("training model...")
            self.fit_model()

        query_wakati= self.parse_text(query)
        target_vec = self.vectorizer.transform([query_wakati])
        sim_list=cosine_similarity(target_vec, self.vec)[0]
        topn_idx=np.array([sim_list]).argsort()[0][::-1][:k]

        text_list=[self.db["text"][idx] for idx in topn_idx]
        path_list=[self.db["fin_path"][idx] for idx in topn_idx]
        sim_list=[sim_list[idx] for idx in topn_idx]

        ret_list=[]
        for i in range(k):
            temp_dict = {}
            temp_dict["path"] = path_list[i]
            temp_dict["text"] = text_list[i].replace(" ", "")
            temp_dict["sim"] = sim_list[i]
            ret_list.append(temp_dict)

        return ret_list