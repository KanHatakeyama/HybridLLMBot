from settings.key import GPT_API_KEY, DEEPL_API_KEY
from LLMSearch.GPTQuery import GPTQuery
from LLMSearch.AnswerBot import AnswerBot
import streamlit as st
from streamlit_chat import message
import os
import json
import csv
import datetime
import pandas as pd


def test_response(res):
    return "good!", ["google.com", "yahoo.com"]


setting_path = 'settings/settings.json'

with open(setting_path) as f:
    settings = json.load(f)


# initiate bot module
bot = AnswerBot(query_module=GPTQuery(GPT_API_KEY),
                DEEPL_API_KEY=DEEPL_API_KEY)
bot.load_model()

log_path = settings["data_path"]+"/chatlog.txt"

st.title("Ask me anything!")

#
gpt4_checkbox_state = st.checkbox('Use GPT4')
reference_checkbox_state = st.checkbox('Load local data')
translate_checkbox_state = st.checkbox('Translate Japanese to English')
show_log_checkbox_state = st.checkbox('Show chat log')

if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []

with st.form("Ask Question"):
    user_message = st.text_area("Input your question")

    submitted = st.form_submit_button("Submit")
    if submitted:

        # show log
        if show_log_checkbox_state:
            st.markdown('Chat log:')
            df = pd.read_csv(log_path, header=None)
            df.columns = ["time", "user", "bot"]
            df.sort_values(by="time", ascending=False, inplace=True)

            for i in range(len(df)):
                st.markdown(df.iloc[i]["time"])
                st.markdown(df.iloc[i]["user"])
                st.markdown(df.iloc[i]["bot"])
                st.markdown("---")

        else:
            # change GPT model
            if gpt4_checkbox_state:
                bot.query_module.model = "gpt-4"
                st.markdown(
                    'GPT-4 enabled. This may take a longer time. (ca. 1 min)')
            else:
                bot.query_module.model = "gpt-3.5-turbo"
                st.markdown('GPT-3.5 enabled. Please wait for ca. 10 sec.')

            if reference_checkbox_state:
                # search for reference data and ask GPT
                st.markdown('Searching for references...')
                k = settings["similarity_top_k"]
                if translate_checkbox_state:
                    answer = bot.ask(user_message, k=k, Ja_to_En=True)
                else:
                    answer = bot.ask(user_message, k=k)
            else:
                # just ask GPT
                answer = bot.query_module.ask_gpt(user_message)

            # log
            with open(log_path, 'a') as f:
                writer = csv.writer(f)
                time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                writer.writerow([time, user_message, answer])

            st.session_state.past.append(user_message)
            st.session_state.generated.append(answer)

            if st.session_state["generated"]:
                for i in range(len(st.session_state.generated) - 1, -1, -1):
                    message(st.session_state.generated[i], key=str(i))
                    message(st.session_state.past[i],
                            is_user=True, key=str(i) + "_user")
