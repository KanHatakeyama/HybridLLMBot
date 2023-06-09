import json
from .DocSplitter import split_documents
from LLMSearch.TextSearcher import TextSearcher
from LLMSearch.ServerEmbedding import ServerEmbedding
from tqdm import tqdm
import glob
from .CleanText import pad_text


class AnswerBot:
    def __init__(self, query_module,
                 DEEPL_API_KEY=None,
                 setting_path='settings/settings.json'):
        with open(setting_path) as f:
            settings = json.load(f)

        self.settings = settings
        self.embedder = ServerEmbedding()
        self.searcher = TextSearcher(self.embedder, self.settings["data_path"])
        self.query_module = query_module
        self.DeepL_API_KEY = DEEPL_API_KEY

        if DEEPL_API_KEY is not None:
            from .DeepLTranslate import DeepLTranslate
            self.translator = DeepLTranslate(self.DeepL_API_KEY)

    def load_model(self):
        self.searcher.load_model()

    def index_documents(self, initiate=False):
        split_documents(initiate=initiate)
        # calc vectors
        chunk_path_list = glob.glob(
            self.settings["data_path"] + "/split/*.txt")

        if not initiate:
            self.load_model()
        count = 0
        print("calculating vectors...")
        for path in tqdm(chunk_path_list):
            self.searcher.calc_text_file(path)
            count += 1

            # 30回に1回保存
            if count % 10 == 0:
                self.searcher.save_model()

        self.searcher.save_model()

    def search_related_documents(self, question, k=3, pad=False):
        if pad:
            question = pad_text(question)
        context_list = self.searcher.search(
            (question, self.settings["chunk_size_limit"]), k)
        return context_list

    def ask(self, question, k=3,
            pad=False,
            text_mode=True,
            Ja_to_En=False):

        if Ja_to_En:
            if self.DeepL_API_KEY is None:
                raise Exception("DeepL_API_KEY is not set.")
            question = self.translator(question)
            #question = "日本語で回答して:"+question
            #print("Japanese to English translation is done.")
            # print(question)

        context_list = self.search_related_documents(question, k, pad)
        ans = self.query_module.reference_ask(question, context_list, k)

        if Ja_to_En:
            eng_ans = "Answer: "+ans["answer"]
            ja_ans = "回答: "+self.translator(ans["answer"], target_lang="JA")
            ans["answer"] = ja_ans + "\n" + eng_ans

        if text_mode:
            return parse_answer(ans)

        return ans


def parse_answer(ans):
    txt = ans["answer"]+"\n\n"
    for i, doc in enumerate(ans["context"]):
        txt += f"Reference {i+1} \n"
        t = doc["text"]
        txt += f"{t} \n"
        p = doc["path"]
        txt += f"{p} \n"
        s = doc["sim"]
        txt += f"Similarity {s}\n"
        txt += "---\n"

    return txt
