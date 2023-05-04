import requests
import numpy as np
import json

class ServerEmbedding:
    def __init__(self,setting_path='settings/settings.json',) -> None:
    
        with open(setting_path) as f:
            settings = json.load(f)
        self.url=settings["EMBED_SERVER_URL"]
        

    def __call__(self,query):

        data = {
            'message': f'{query}'
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.post(self.url, json=data, headers=headers)

        v=response.json()["response"]
        v=np.array(v)
        return v.reshape(1,-1).astype('float32')