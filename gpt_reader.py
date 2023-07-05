from LLMSearch.AnswerBot import AnswerBot, parse_answer
from LLMSearch.GPTQuery import GPTQuery
from settings.key import GPT_API_KEY, DEEPL_API_KEY
import openai
import streamlit
from LLMSearch.BM25VecDB import BM25VecDB


openai.api_key = GPT_API_KEY

#gpt=GPTQuery(GPT_API_KEY,model="gpt-3.5-turbo-16k")
gpt=GPTQuery(GPT_API_KEY,model="gpt-3.5-turbo")
gpt4=GPTQuery(GPT_API_KEY,model="gpt-4-0613")

bot1 = AnswerBot(query_module=GPTQuery(GPT_API_KEY,
                                       # model="gpt-4-0613"
                                        ),
                searcher=BM25VecDB(),
                DEEPL_API_KEY=DEEPL_API_KEY,
                )

input_text = streamlit.text_input('検索キーワードを入力')
input_text2 = streamlit.text_input('GPTに処理させる内容を入力',"1.分子構造 2.物性 を教えて")
k=streamlit.number_input('解析件数',min_value=1, max_value=100, value=5, step=1)
#実行ボタン



if len(input_text) > 0 and len(input_text2) > 0:
    # get references
    related_documents=[]
    for bot in [bot1]:
        related_documents+=bot.search_related_documents(input_text, k=10**3)

    n_hits=len(related_documents)
    related_documents=related_documents[:k]

    result_area = streamlit.empty()
    text = ''
    text += f"## {n_hits}件の関連文献が見つかりました\n"


    #get titles
    for idx in (range(len(related_documents))):
        lit_text=related_documents[idx]["text"]
        path=related_documents[idx]["path"]

        with open(path, 'r') as f:
            title= f.readline().strip()
            
        related_documents[idx]["title"]=title

    text+="## GPTの回答は...\n"        
    result_area.write(text)


    # GPT ans
    for idx in (range(len(related_documents))):
        lid=idx+1
        title=related_documents[idx]["title"]
        lit_text=related_documents[idx]["text"]

        text+=f"\n### 文献 {lid}\n"
        text+=f"#### {title}\n"
        text+=f"- {lit_text}\n"
        text+="#### GPT3解説\n"

        completion=gpt.ask_gpt(input_text2,context_text=f"title: {title} content: {lit_text}",stream=True)
        for chunk in completion:
            next = chunk['choices'][0]['delta'].get('content', '')
            text += next
            result_area.write(text)


    #総括
    all_context=[{"text":input_text2+": "+ text[text.find("## GPTの回答は"):]}]
    text+="\n### 総括 by GPT4 \n"
    completion=gpt4.ask_gpt("要約して: "+input_text2,context_text= text[text.find("## GPTの回答は"):][:8000],stream=True)
    for chunk in completion:
        next = chunk['choices'][0]['delta'].get('content', '')
        text += next
        result_area.write(text)