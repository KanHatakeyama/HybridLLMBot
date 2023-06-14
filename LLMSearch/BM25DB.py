from . DocSplitter import clean_text, split_text
import json
import MeCab
import re
import sqlite3
import unicodedata
mecab = MeCab.Tagger("-Owakati")


def remove_symbols(text):
    return ''.join(ch for ch in text if not unicodedata.category(ch).startswith('P'))



class BM25DB:
    def __init__(self,
                 setting_path='settings/settings.json',
                 parser=mecab,
                 ) -> None:

        with open(setting_path) as f:
            settings = json.load(f)

        self.db_path = settings["data_path"]+"/meta/text_bm25.db"
        self.chunk_size_limit = settings["chunk_size_limit"]

        self.parser= parser
        self.conn = sqlite3.connect(self.db_path)
        self.c = self.conn.cursor()

    def initiate(self):
        self.c.execute("DROP TABLE IF EXISTS docs;")
        self.c.execute("CREATE VIRTUAL TABLE docs USING fts5(content, filepath, number);")
        self.add_record("test_path", "test text", 0, commit=True)

    def parse_text(self, text):
        return self.parser.parse(text).strip()

    def add_record(self, path, text, split_id,commit=True):
        doc_wakati = self.parse_text(text)
        self.c.execute("INSERT INTO docs(content, filepath, number) VALUES(?, ?, ?);", (doc_wakati, path, split_id))
        if commit:
            self.commit()

    def commit(self):
        self.conn.commit()

    def add_text(self, path):
        self.c.execute("SELECT filepath FROM docs;")
        finished_list= self.c.fetchall()
        finished_list= [fp[0] for fp in finished_list]
        if path in finished_list:
            print(f"{path} already added")
            return

        with open(path, 'r') as f:
            text = f.read()

        text = clean_text(text)

        chunk_list = split_text(text, self.chunk_size_limit)
        for i, chunk in enumerate(chunk_list):
            self.add_record(path, chunk, i, commit=False)
        self.commit()

    def search(self, query, k=10):
        query= remove_symbols(query)
        # 単語ごとにクエリを作成

        query_wakati= self.parse_text(query)
        query_words = query_wakati.split()

        # 各単語で検索を行い、結果を組み合わせる
        combined_results = []
        for query_word in query_words:
            #c.execute("SELECT content, bm25(docs) FROM docs WHERE docs MATCH ? ORDER BY bm25(docs);", (query_word,))
            self.c.execute("SELECT content, filepath, bm25(docs) FROM docs WHERE docs MATCH ? ORDER BY bm25(docs);", (query_word,))
            results = self.c.fetchall()
            combined_results.extend(results)

        # 各文書のスコアを合計
        score_dict = {}
        for result in combined_results:
            if result[0] in score_dict:
                score_dict[result[0]] += result[2]
            else:
                score_dict[result[0]] = result[2]

        # 結果をスコアでソート
        sorted_results = sorted(score_dict.items(), key=lambda x: x[1])

        ret_list=[]
        for result in sorted_results:
            key=result[0]
            for record in combined_results:
                if record[0]==key:
                    break
            temp_dict = {}
            temp_dict["path"] = record[1]
            temp_dict["text"] = record[0]
            temp_dict["sim"] = record[2]
            ret_list.append(temp_dict)

        return ret_list[:k]