import requests


class DeepLTranslate:
    def __init__(self, API_KEY,
                 server_url="https://api-free.deepl.com/v2/translate") -> None:
        self.API_KEY = API_KEY
        self.server_url = server_url

    def __call__(self, text, target_lang="EN"):
        result = requests.get(
            self.server_url,
            params={
                "auth_key": self.API_KEY,
                "target_lang": target_lang,
                "text": text,
            },
        )

        translated_text = result.json()["translations"][0]["text"]
        return translated_text
