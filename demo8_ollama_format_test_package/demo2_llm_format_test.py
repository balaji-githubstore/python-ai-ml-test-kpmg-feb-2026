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

actual_cleared_data=re.sub(pattern=r"^```(?:json)?|```$",repl="",string=actual_llm_output,flags=re.MULTILINE).strip()
print(actual_cleared_data)

# json object --> make sure it is in proper json format
data=json.loads(actual_cleared_data)

# pass list or dic --> here passing list of json object
df=pd.DataFrame([data])

print(df)

# {'age', 'city', 'name'}
print(set(df.columns))

# testcase fails if columns are not present
expected_columns={'age', 'city', 'name'}
assert set(df.columns)==expected_columns, "Output does not match with expected_columns"

# print(df.isnull().sum().sum())

assert df.isnull().sum().sum()==0, "Output has missing value is there"


"""
Example 
case 1: follow system prompt strictly 
case 2: tries to follow user prompt
case 3: slightly malformed/mixed output
case 4: completely ignore system

"""