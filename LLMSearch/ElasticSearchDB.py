from elasticsearch import Elasticsearch
import datetime
import json
from transformers import BertJapaneseTokenizer
from . DocSplitter import clean_text, split_text
import nltk
nltk.download('punkt')

# elasticsearch settings
url = "http://localhost:9200"
es = Elasticsearch(url)
index_name = "gpt_search_index"

if es.ping():
    print('Elasticsearch is running')
else:
    print('Elasticsearch could not be reached!')


class ElasticSearchDB:
    def __init__(self,
                 setting_path='settings/settings.json',
                 initiate=False) -> None:

        with open(setting_path) as f:
            settings = json.load(f)

        self.db_path = settings["data_path"]+"/meta/text.db"
        self.chunk_size_limit = settings["chunk_size_limit"]

        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            settings["JAPANESE_TOKENIZER_MODEL"])

        if initiate:
            try:
                es.indices.delete(index=index_name, ignore=[400, 404])
            except:
                pass
            # es.create_index(index_name)
            es.indices.create(
                index=index_name,
                body=japanese_analyzer_body)

    def add_record(self, path, text, split_id):
        doc = {
            'author': "test",
            'text': text,
            "path": path,
            "split_id": split_id,
            'timestamp': datetime.datetime.now(),
        }
        res = es.index(index=index_name, id=path, body=doc)

    def add_text(self, path):
        # check existing data
        res = exact_match_search("path", path)
        hit = res['hits']['total']['value']
        if hit > 0:
            # print(f"{path} already added")
            return

        # open text
        with open(path, 'r') as f:
            text = f.read()
        text = clean_text(text)

        # split
        chunk_list = split_text(text, self.chunk_size_limit)
        for i, chunk in enumerate(chunk_list):
            self.add_record(path, chunk, i)

    def tokenize_query(self, query):
        # prepare tokenized list in japanese and english
        wakati_ids = self.tokenizer.encode(query, return_tensors='pt')
        tokenized_list = self.tokenizer.convert_ids_to_tokens(
            wakati_ids[0].tolist())
        eng_tokenized_list = nltk.word_tokenize(query)

        tokenized_list.extend(eng_tokenized_list)
        tokenized_list = [i for i in tokenized_list if i not in [
            "[UNK]", "[SEP]", "[CLS]", "[PAD]"]]
        tokenized_list = [i for i in tokenized_list if i.find("##") == -1]
        tokenized_list = list(set(tokenized_list))
        return tokenized_list


def search(body):
    return es.search(index=index_name, body=body)


def exact_match_search(field, query):
    body = {
        "query": {
            "term": {
                f"{field}.keyword": query
            }
        }
    }

    return search(body)


def keyword_search(query_list, field="text"):
    should_list = []
    for query in query_list:
        should_list.append({"match": {field: query}})

    res = es.search(index=index_name, body={
        "query": {
            "bool": {
                "should":   should_list,
            }
        }
    })

    return res


japanese_analyzer_body = {
    "settings": {
        "analysis": {
            "analyzer": {
                "my_analyzer": {
                    "type": "kuromoji"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "field_name": {
                "type": "text",
                "analyzer": "my_analyzer"
            }
        }
    }
}
