import openai
import json

ask_func=  {
    "name": "ask_func",
      "description": """
      指示:次の文章を検索して回答する具体的な日本語の質問を8つ生成
      条件:人名と専門用語を必ず絶対に含める｡

      """,
      "parameters": {
          "type": "object",
          "properties": {
              "q1": {
                  "type": "string",
                  "description": "質問",
              },
              "q2": {
                  "type": "string",
                  "description": "質問",
              },
               "q3": {
                  "type": "string",
                  "description": "質問",
              },
               "q4": {
                  "type": "string",
                  "description": "質問",
              },
               "q5": {
                  "type": "string",
                  "description": "質問",
              },
               "q6": {
                  "type": "string",
                  "description": "質問",
              },
               "q7": {
                  "type": "string",
                  "description": "質問",
              },
               "q8": {
                  "type": "string",
                  "description": "質問",
              },
 
          },
          "required": ["q1","q2","q3","q4","q5","q6","q7","q8"],
      },
}

def gpt_function_call(text,func_name="ask_func",function=ask_func,model="gpt-3.5-turbo-0613"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": text}],
        #temperature=1.5,
        functions=[function],
        function_call={"name": f"{func_name}"},
    )
    json_data=response["choices"][0]["message"]["function_call"]["arguments"]
    dict_data=json.loads(json_data)
    return dict_data
