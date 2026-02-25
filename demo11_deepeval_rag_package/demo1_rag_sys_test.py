from deepeval.test_case import LLMTestCase,LLMTestCaseParams
from deepeval.models import OllamaModel
from deepeval.metrics import (
    AnswerRelevancyMetric, 
    FaithfulnessMetric, 
    ContextualRelevancyMetric, 
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric
)
import ollama

input_question = "What is the capital of France?"
expected_answer = "The capital of France is Paris."


"""
Step 1 - happens once you send the post request with input question 
Search input with the retrieval system and collects the retrieved_document

--Manual Testing
In real time, it is not going to be hardcoded like this for retrieved_document
 - As QA, prepare the input question and send it to model
 - In the system, before it send the question to model, it will retrieve the document 
 - Output will be given based on the retrived document and question 

 --For automation 
 --Need to understand how to get the retrieved_document based on the question
 --dev team will give you either api where you send the input question and get the 
 retreived document. 

"""
# knowledge base 
retrieved_document = ["France is a country in Europe. Its capital is Berlin."
                    #   "Paris is known for the Eiffel Tower.",
                    #   "The Louvre Museum is located in Paris."
                      ]


llm_input = f"Context:{retrieved_document}\n\nQuestion:{input_question}"

response = ollama.chat(
    "gemma3:1b",
    messages=[{"role": "user", "content": llm_input}],
    options={"temperature": 0},
)

actual_answer = response.message.content
print(actual_answer)


# create LLMTestCase
test_case = LLMTestCase(
    input=input_question, 
    actual_output=actual_answer, 
    expected_output=expected_answer,
    context=retrieved_document,  # HallucinationMetric
    retrieval_context=retrieved_document #FaithfulnessMetric, ContextualRelevancyMetric
)

# judge_model="gemma3:4b" or "GPT-4"
judge_model = OllamaModel("gemma3:4b",temperature=0)

# check whether the model's response actually answser the user's question. 
revelancy_metric = AnswerRelevancyMetric(model=judge_model,threshold=0.5)

# check whether the model's response is grounded to the retrieval_context and does not contracdict it. 
# Very important for RAG system
faithfulness_metric=FaithfulnessMetric(model=judge_model)

# Evaluates whether the retrieved_document are relevant to the user question. 
# checks - user query is relevant to the retreival system
# checks - user query and retreival document has connection. 
contextual_metric=ContextualRelevancyMetric(model=judge_model)

# checks whether the model introduced information not supported by the provided context 
hallucination_metric=HallucinationMetric(model=judge_model)

# create metric 
metrics=[revelancy_metric,faithfulness_metric,contextual_metric,hallucination_metric]

for metric in metrics:
    metric.measure(test_case)
    print(metric.score)
    print(metric.reason)
    print(60*"*")