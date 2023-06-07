from sqlite_utils import Database
from . DocSplitter import clean_text, split_text
import glob
from tqdm import tqdm
import json
from transformers import BertJapaneseTokenizer
# path_listでcountを取る
from collections import Counter
import nltk
nltk.download('punkt')


class SQLTextDB:
    def __init__(self,
                 setting_path='settings/settings.json',
                 initiate=False) -> None:

        with open(setting_path) as f:
            settings = json.load(f)

        self.db_path = settings["data_path"]+"/meta/text.db"
        self.chunk_size_limit = settings["chunk_size_limit"]

        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            settings["JAPANESE_TOKENIZER_MODEL"])

        self.db = Database(self.db_path, recreate=initiate)
        if initiate:
            self.add_record("path", "text", 0)
            self.db["text"].enable_fts(["text"], tokenize="porter")
            self.db["finished"].enable_fts(["path"], tokenize="porter")

    def add_record(self, path, text, split_id):
        self.db['text'].insert({
            "path": path,
            'split_id': split_id,
            'text': text
        })
        if split_id == 0:
            self.db['finished'].insert({"path": path})

    def add_text(self, path):
        finished_list = list(self.db["finished"].rows)
        finished_list = [i["path"] for i in finished_list]
        if path in finished_list:
            print(f"{path} already added")
            return

        with open(path, 'r') as f:
            text = f.read()

        text = clean_text(text)

        chunk_list = split_text(text, self.chunk_size_limit)
        for i, chunk in enumerate(chunk_list):
            self.add_record(path, chunk, i)

    def tokenize_query(self, query):
        # prepare tokenized list in japanese and english
        wakati_ids = self.tokenizer.encode(query, return_tensors='pt')
        tokenized_list = self.tokenizer.convert_ids_to_tokens(
            wakati_ids[0].tolist())
        #tokenized_list = tokenized_list[1:-1]
        eng_tokenized_list = nltk.word_tokenize(query)

        tokenized_list.extend(eng_tokenized_list)
        tokenized_list = [i for i in tokenized_list if i not in [
            "[UNK]", "[SEP]", "[CLS]", "[PAD]"]]
        tokenized_list = [i for i in tokenized_list if i.find("##") == -1]
        tokenized_list = list(set(tokenized_list))
        return tokenized_list

    def search_text(self, query, k=1000):
        tokenized_list = self.tokenize_query(query)

       # search words
        path_list = []
        for key in tokenized_list:
            rows = self.db.execute(
                "SELECT * FROM text WHERE text LIKE ?", (f'%{key}%',))

            temp_path_list = [row[:] for row in rows]
            path_list.extend(temp_path_list)

        c = Counter(path_list)
        common_list = c.most_common(k)
        return common_list
