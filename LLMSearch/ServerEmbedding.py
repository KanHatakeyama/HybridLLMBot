import requests
import numpy as np

class ServerEmbedding:
    def __init__(self,url="http://localhost:5000/api") -> None:
        self.url=url
        

    def __call__(self,query):

        data = {
            'message': f'{query}'
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.post(self.url, json=data, headers=headers)

        v=response.json()["response"]
        v=np.array(v)
        return v.reshape(1,-1).astype('float32')