# pip install ollama
import ollama


response = ollama.chat(model="gemma3:1b", 
                       messages=[
                               {
                                   'role': 'system', 
                                    'content': 'you are a json generator. Always produce output in the format: {"name":"<string>","age":<integer>, city: "<string>"}'
                               },
                               {
                                   'role':'user',
                                   'content':'Generate json with name, age, city'
                               }

                           ])

actual_llm_output=response.message.content
print(actual_llm_output)

 