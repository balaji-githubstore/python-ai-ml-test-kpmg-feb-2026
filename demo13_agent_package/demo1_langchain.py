# pip install langchain langchain-community ollama langchain-ollama

from langchain_ollama import ChatOllama
from langchain.agents import create_agent

# Langchain LLM 
llm=ChatOllama(model="gemma3:1b",temperature=0)


# register agent with langchain and use it to get response 
agent=create_agent(model=llm)

# agent invoke 
response=agent.invoke({"messages":[{"role":"user","content":"what is the captial of india?"}]})

"""
messages[0] --> human
messages[1]--> latest AI response 

or to get message use -1
messages[-1]--> latest AI response
"""
# access agent response to get AI message
actual_result=response["messages"][-1].content
print(actual_result)


# as usual conduct format evaluation, metrics, deepeval testing 