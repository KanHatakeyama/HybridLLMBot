from flask import Flask, jsonify, request
import json
import numpy as np
from LLMSearch.legacy.RWKVEmbedding import RWKVEmbedding
from LLMSearch.VicunaEmbedding import VicunaEmbedding

with open('settings/settings.json') as f:
    settings = json.load(f)

app = Flask(__name__)

embedder=VicunaEmbedding()
#embedder=RWKVEmbedding(settings)

@app.route('/api', methods=['POST'])
def api():
    data = request.get_json()
    message = data.get('message', 'No message provided')
    vec=embedder(message).reshape(-1)
    #vec=np.array([1,2,3])
    #vec = [1, 2, 3, 4, 5]
    response = {'response':vec.tolist()}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False)
