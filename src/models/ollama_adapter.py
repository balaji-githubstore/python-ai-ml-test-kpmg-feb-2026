import requests


class OllamaAdapter:

    def generate(self, prompt):
        # url, model, taken from json or any other external file
        end_point = "http://localhost:11434/api/chat"
        response = requests.post(
            url=end_point,
            json={
                "model": "gemma3:1b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        ).json()
        return response["message"]["content"]
