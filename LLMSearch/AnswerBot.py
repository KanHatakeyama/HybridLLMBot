import json
#from .DocSplitter import split_documents
#from .TextSearcher import TextSearcher
#from .ServerEmbedding import ServerEmbedding
from .SQLTextDB import SQLTextDB
#from .BM25DB import BM25DB
from tqdm import tqdm
import glob
#from .CleanText import pad_text


class AnswerBot:
    def __init__(self, query_module,
                    searcher,
                 DEEPL_API_KEY=None,
                 setting_path='settings/settings.json'):
        with open(setting_path) as f:
            settings = json.load(f)

        self.settings = settings
        #self.embedder = ServerEmbedding()
        #self.searcher = TextSearcher(self.embedder, self.settings["data_path"])
        self.searcher = searcher
        self.setting_path = setting_path
        self.query_module = query_module
        self.DeepL_API_KEY = DEEPL_API_KEY
        self.file_url = settings["FILE_URL"]

        if DEEPL_API_KEY is not None:
            from .DeepLTranslate import DeepLTranslate
            self.translator = DeepLTranslate(self.DeepL_API_KEY)

    # def load_model(self):
    #    self.searcher.load_model()

    def index_documents(self, initiate=False):

        if initiate:
            self.searcher.initialize()

        path_list = glob.glob(
            self.settings["data_path"]+"/original/**/*.txt", recursive=True)

        for path in tqdm(path_list):
            self.searcher.add_text(path)

    def search_related_documents(self, question, k=3):
        context_list = self.searcher.search(question, k=k)
        return context_list

    def ask(self, question,
            k=3,
            context_list=None,
            pad=False,
            text_mode=True,
            Ja_to_En=False,
            stream=False):

        if Ja_to_En:
            if self.DeepL_API_KEY is None:
                raise Exception("DeepL_API_KEY is not set.")
            question = self.translator(question)


        if context_list is None:
            context_list = self.search_related_documents(question, k, pad)

        k = min(k, len(context_list))
        ans = self.query_module.reference_ask(
            question, context_list, k, stream=stream)

        if Ja_to_En:
            eng_ans = "Answer: "+ans["answer"]
            ja_ans = "回答: "+self.translator(ans["answer"], target_lang="JA")
            ans["answer"] = ja_ans + "\n" + eng_ans

        if stream:
            text_mode = False
        if text_mode:
            return parse_answer(ans)

        return ans


def parse_answer(ans, text_length=100, base_url="http://localhost:8099/"):
    txt = ans["answer"]+"\n\n"
    for i, doc in enumerate(ans["context"]):
        txt += f"- Reference {i+1} \n"
        p = doc["path"]
        p = base_url+p
        txt += f"{p} \n"
        s = doc["sim"]
        txt += f"Similarity {s}\n"
        t = doc["text"][:text_length]
        txt += f"{t} \n"
        txt += "***\n"

    return txt
