import openai


class GPTQuery:
    def __init__(self,API_KEY,model="gpt-3.5-turbo") -> None:
        openai.api_key = API_KEY
        self.model=model



    def ask_gpt(self,query,context_text):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "assistant", "content": context_text},
                {"role": "user", "content": query},
            ],
        )
        return (response.choices[0]["message"]["content"].strip())

    def reference_ask(self,query,context):
        res_dict={}
        res_dict["answer"]=self.ask_gpt(query,context["text"])
        res_dict["reference"]=context["path"]
        res_dict["sim"]=context["sim"]

        with open(context["path"]) as f:
            res_dict["context"]=f.read()

        return res_dict


