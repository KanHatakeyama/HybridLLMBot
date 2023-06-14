from LLMSearch.AnswerBot import AnswerBot, parse_answer
from LLMSearch.GPTQuery import GPTQuery
from settings.key import GPT_API_KEY, DEEPL_API_KEY
import openai
import streamlit
from LLMSearch.Embedding.SBERTFineTuneEmbedding import SBERTFineTuneEmbedding
from LLMSearch.VectorSearcher import VectorSearcher
from LLMSearch.BM25DB import BM25DB
from LLMSearch.SQLTextDB import SQLTextDB
import json
embedder=SBERTFineTuneEmbedding()
searcher=VectorSearcher(embedder)


openai.api_key = GPT_API_KEY

#bot = AnswerBot(query_module=GPTQuery(GPT_API_KEY),
#                searcher=searcher,
#                DEEPL_API_KEY=DEEPL_API_KEY)
bot1 = AnswerBot(query_module=GPTQuery(GPT_API_KEY),
                searcher=searcher,
                DEEPL_API_KEY=DEEPL_API_KEY,
                )

bot2=AnswerBot(query_module=GPTQuery(GPT_API_KEY),
                searcher=SQLTextDB(), 
                DEEPL_API_KEY=DEEPL_API_KEY,
                )

bot3=AnswerBot(query_module=GPTQuery(GPT_API_KEY),
                searcher=BM25DB(),
                DEEPL_API_KEY=DEEPL_API_KEY,   
                )

input_text = streamlit.text_input('質問を入力')

if len(input_text) > 0:
    result_area = streamlit.empty()
    text = ''

    text += "## 以下の関連文献が見つかりました\n"

    # get references
    related_documents=[]
    for bot in [bot1,bot2,bot3]:
        related_documents+=bot.search_related_documents(input_text, k=10)

    #integrate references
    related_documents=[{"path":d["path"],"text":d["text"],"sim":0} for d in related_documents]
    list_of_json = [json.dumps(d, sort_keys=True) for d in related_documents]
    related_documents= list(map(json.loads, set(list_of_json)))

    ans = {}
    ans["answer"] = ""
    ans["context"] = related_documents
    str_ans = parse_answer(ans, base_url=bot.file_url,text_length=200)
    text += str_ans

    text += "## GPTの回答は以下の通りです\n"
    result_area.write(text)

    # GPT ans

    related_documents_list = [related_documents[i:i+5]
                              for i in range(0, len(related_documents), 5)]
    for i, related_documents in enumerate(related_documents_list):
        start = i*5+1
        end = (i+1)*5

        text += f"\n### Reference{start}から{end}をもとにした回答\n"
        gpt_ans = bot.ask(
            input_text, context_list=related_documents, stream=True)
        completion = gpt_ans["answer"]
        for chunk in completion:
            next = chunk['choices'][0]['delta'].get('content', '')
            text += next
            result_area.write(text)
