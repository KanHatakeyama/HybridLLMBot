from flask import Flask, jsonify, request
import json
import numpy as np
#from LLMSearch.legacy.RWKVEmbedding import RWKVEmbedding
from LLMSearch.GPTEmbedding import GPTEmbedding

with open('settings/settings.json') as f:
    settings = json.load(f)

app = Flask(__name__)

if settings['EMBED_MODE'] == 'GPT':
    from settings.key import GPT_API_KEY
    embedder = GPTEmbedding(GPT_API_KEY)
elif settings['EMBED_MODE'] == 'Vicuna':
    from LLMSearch.VicunaEmbedding import VicunaEmbedding
    embedder = VicunaEmbedding()
else:
    raise ValueError('EMBED_MODE not recognized:', settings['EMBED_MODE'])
# embedder=RWKVEmbedding(settings)


@app.route('/api', methods=['POST'])
def api():
    data = request.get_json()
    message = data.get('message', 'No message provided')
    vec = embedder(message).reshape(-1)
    response = {'response': vec.tolist()}

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=False)
