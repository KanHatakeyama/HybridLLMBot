import streamlit as st
from streamlit_chat import message
import os
import json

def parse_response(res):
    return "good!",["google.com","yahoo.com"]
setting_path='settings/settings.json'

with open(setting_path) as f:
    settings = json.load(f)

from LLMSearch.AnswerBot import AnswerBot
from LLMSearch.GPTQuery import GPTQuery
from settings.key import API_KEY

#initiate bot module
bot=AnswerBot(query_module=GPTQuery(API_KEY))
bot.load_model()

log_path=settings["data_path"]+"/chatlog.txt"

st.title("Ask me anything!")

if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []

with st.form("Ask Question"):
    user_message = st.text_area("Input your question")

    submitted = st.form_submit_button("Submit")
    if submitted:
        #res = searcher.query(user_message)
        answer=bot.ask(user_message)

        with open(log_path, "a") as f:
            f.write(f"{user_message}\n{answer}\n\n\n")

        st.session_state.past.append(user_message)
        st.session_state.generated.append(answer)

        if st.session_state["generated"]:
            for i in range(len(st.session_state.generated) - 1, -1, -1):
                message(st.session_state.generated[i], key=str(i))
                message(st.session_state.past[i],
                        is_user=True, key=str(i) + "_user")

                #for url in url_list:
                #    http_url = "http://"+url
                #    st.markdown(f"[{http_url}]({http_url})")
