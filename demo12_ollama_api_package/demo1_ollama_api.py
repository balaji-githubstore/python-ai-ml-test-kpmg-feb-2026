import requests

# base_url="http://localhost:11434"
# resource="/api/chat"

end_point = "http://localhost:11434/api/chat"

response=requests.post(url=end_point,
              json={
                    "model": "gemma3:1b", 
                    "messages": [{"role": "user","content":"Summarize: cat is sitting on the mat"}],
                    "stream":False
                    }
              )

response=response.json()

actual_result=response["message"]["content"]
print(actual_result)
print(response["model"])
print(response["eval_count"])
print(response["total_duration"])
print(f"Total duration {response["total_duration"]/1e9}")
print(f"Total duration {response["prompt_eval_duration"]/1e9}")