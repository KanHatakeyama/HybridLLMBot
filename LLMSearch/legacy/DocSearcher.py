# https://dev.classmethod.jp/articles/get-reference-in-query-of-llamaindex/
import json
import os
from typing import List

from llama_index import Document, GPTSimpleVectorIndex, ServiceContext, SimpleDirectoryReader
from llama_index.data_structs.node_v2 import DocumentRelationship
from llama_index.response.schema import RESPONSE_TYPE
from llama_index import LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI

import json
settings = json.load(open("data/settings.json", "r"))


def parse_response(response: RESPONSE_TYPE):
    url_list = []
    res = ""
    res += (response.response)+"\n"
    res += ("===================\n")
    for node in response.source_nodes:
        if node.node.extra_info is not None:
            if "file_name" in node.node.extra_info:
                file_path = node.node.extra_info["file_name"]

                file_path = file_path.replace("\\", "/")
                file_path = file_path.replace("data/documents/", "")
                file_path = settings["url"]+"/"+file_path
                url_list.append(file_path)
                res += (file_path)+"\n"
        res += "文献中の位置: "
        res += str(node.node.node_info)+"\n"
        if node.score is not None:
            res += "文献データとの類似度: " + str(node.score)+"\n"
        res += ("===================\n")
        res += "  [関連の深い文献]\n"
        res += str(node.node.text)+"\n"
        res += ("===================\n")

    return res, url_list


class DocSearcher:
    def __init__(self, setting_path):
        self.settings = json.load(open(setting_path, "r"))
        data_path = self.settings["data_path"]
        chunk_size_limit = self.settings["chunk_size_limit"]

        folderpath_index = os.path.join(data_path, "indexes")
        folderpath_documents = os.path.join(data_path, "documents")
        filepath_cache_metadata = os.path.join(
            folderpath_index, "metadata.json")
        filepath_cache_index = os.path.join(folderpath_index, "index.json")

        self.folderpath_documents = folderpath_documents
        self.folderpath_index = folderpath_index
        self.filepath_cache_metadata = filepath_cache_metadata
        self.filepath_cache_index = filepath_cache_index
        self.chunk_size_limit = chunk_size_limit

        self.init_directory_reader()

    def init_directory_reader(self):
        def filename_fn(filename): return {"file_name": filename}

        self.directory_reader = SimpleDirectoryReader(
            self.folderpath_documents,
            file_metadata=filename_fn,
            recursive=True,
        )

    def update_index(self):
        self.init_directory_reader()
        # select docments
        filenames = [str(input_file)
                     for input_file in self.directory_reader.input_files]
        filenames_ = []

        # check indexed and unindexed files
        if os.path.exists(self.filepath_cache_metadata):
            metadata = json.loads(
                open(self.filepath_cache_metadata, "r").read())
            filenames_ = metadata["filenames"]

        # read documents
        self.documents = self.directory_reader.load_data()

        prompt_helper = PromptHelper(max_input_size=3000,
                                     num_output=300,
                                     max_chunk_overlap=20,
                                     chunk_size_limit=self.chunk_size_limit)
        service_context = ServiceContext.from_defaults(
            llm_predictor=LLMPredictor(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=3000)),
            chunk_size_limit=self.chunk_size_limit,
            prompt_helper=prompt_helper,
        )

        save_flag = False

        # make index
        # index all for the first scan
        if len(self.documents) == 0 or len(filenames) != len(filenames_):
            self.index = GPTSimpleVectorIndex.from_documents(
                self.documents, service_context=service_context)
            print("first scan. index all files", self.documents)
            save_flag = True
        else:
            # load index from cache
            self.index = GPTSimpleVectorIndex.load_from_disk(
                self.filepath_cache_index)

            # update index for additional files
            unindexed_filenames = set(filenames) - set(filenames_)
            unindexed_filenames = list(unindexed_filenames)

            for doc in self.documents:
                if doc.extra_info["file_name"] in unindexed_filenames:
                    print("add index for ", doc)
                    self.index.insert(doc)
                    save_flag = True

        # save cache
        if save_flag:
            os.makedirs(self.folderpath_index, exist_ok=True)
            self.index.save_to_disk(self.filepath_cache_index)
            json.dump({"filenames": filenames}, open(
                self.filepath_cache_metadata, "w"), indent=2)

            # print("index saved to disk", self.documents)

    def query(self, query, similarity_top_k=None):
        if similarity_top_k is None:
            similarity_top_k = self.settings["similarity_top_k"]
        response = self.index.query(query,
                                    similarity_top_k=similarity_top_k)

        return response
