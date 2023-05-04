import openai


class GPTQuery:
    def __init__(self,API_KEY,model="gpt-3.5-turbo") -> None:
        openai.api_key = API_KEY
        self.model=model


    def ask_gpt(self,query,context_text=None):
        if context_text is not None:
            messages=[
                {"role": "assistant", "content": context_text},
                {"role": "user", "content": query},
            ]
        else:
            messages=[
                {"role": "user", "content": query},
            ]

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
        )
        return (response.choices[0]["message"]["content"].strip())

    def reference_ask(self,query,context_list,k=2):
        context_text=""
        for i in range(k):
            context_text+=context_list[i]["text"]+"."


        res_dict={}
        res_dict["answer"]=self.ask_gpt(query,context_text)
        res_dict["context"]=context_list[:k]


        return res_dict


