import openai


class GPTQuery:
    def __init__(self, API_KEY, model="gpt-3.5-turbo") -> None:
        openai.api_key = API_KEY
        self.model = model

    def ask_gpt(self, query, context_text=None, stream=False):
        if context_text is not None:
            messages = [
                {"role": "assistant", "content": context_text},
                {"role": "user", "content": "Answer question from given context. Never use fictitious information: "+query},
            ]
        else:
            messages = [
                {"role": "user", "content": query},
            ]

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            stream=stream,
        )
        if not stream:
            return (response.choices[0]["message"]["content"].strip())
        else:
            return response

    def reference_ask(self, query, context_list, k=2, ref_max_length=3500, stream=False):
        context_text = ""
        for i in range(k):
            context_text += context_list[i]["text"]+"."

        context_text = context_text[:ref_max_length]

        res_dict = {}
        res_dict["answer"] = self.ask_gpt(query, context_text, stream=stream)
        res_dict["context"] = context_list[:k]

        return res_dict
