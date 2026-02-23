# pip install ollama
import ollama
import json
import pandas as pd
import re

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

actual_cleared_data=re.sub(r"^```(?:json)?|```$","",actual_llm_output,flags=re.MULTILINE).strip()
print(actual_cleared_data)

# json object --> make sure it is in proper json format
data=json.loads(actual_cleared_data)

# pass list or dic --> here passing list of json object
df=pd.DataFrame([data])

print(df)

print(df.columns)

# will start at 
