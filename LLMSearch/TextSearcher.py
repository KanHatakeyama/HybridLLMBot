import faiss
import joblib
import os


class TextSearcher:
    def __init__(self, embedder, base_path) -> None:
        self.embedder = embedder
        self.base_path = base_path
        self.path_list = []
        dim = self.embedder("a").shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexFlatIP(dim)

        meta_path = self.base_path+"/meta"
        if not os.path.exists(meta_path):
            os.mkdir(meta_path)

    def calc_text_file(self, path):
        if path in self.path_list:
            return

        with open(path, 'r') as f:
            text = f.read()
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
