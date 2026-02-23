import ollama


response = ollama.chat(model="gemma3:1b", messages=[{'role': 'user', 'content': 'Summarize: The cat sat on the mat'}])

print(response.message.content)    

""" 

"""