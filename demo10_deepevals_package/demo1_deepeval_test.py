# pip install deepeval
"""
Using DeepEval Framework to evaluate the model output.
Calculate

1. AnswserRelevancyMetric --> check if the answer is relevant to the question. 
    Not validate with the expected answer. 
    Just check if the answer is relevant to the question or not.


2. Correctness using GEval --> check if the answer is correct or not.
LLM-as-a-judge --> we can use "gemma3:4b" or "GPT-4" as a judge model.

"""

import ollama
from deepeval.test_case import LLMTestCase,LLMTestCaseParams
from deepeval.models import OllamaModel
from deepeval.metrics import AnswerRelevancyMetric, GEval, BiasMetric, ToxicityMetric


input_question = "What is the capital of France?"
expected_answer = "The capital of France is Paris."

# testing model "gemma3:1b" - output 
response = ollama.chat(
    "gemma3:1b",
    messages=[{"role": "user", "content": input_question}],
    options={"temperature": 0},
)

actual_answer = response.message.content
print(actual_answer)


# create LLMTestCase
test_case = LLMTestCase(
    input=input_question, 
    actual_output=actual_answer, 
    expected_output=expected_answer
)

# judge_model="gemma3:4b" or "GPT-4"
judge_model = OllamaModel("gemma3:4b",temperature=0)


# calculate AnswerRelevancyMetric using LLM-as-a-judge (judge_model) - not validate with expected_output
revelancy_metric = AnswerRelevancyMetric(model=judge_model,threshold=0.5)

revelancy_metric.measure(test_case)
print(f"Answer Relevancy Score: {revelancy_metric.score}")
print(f"Reason:{revelancy_metric.reason}")

# calculate AnswerCorrectness using GEval
correctness_metric=GEval(
    name="Correctness",
    model=judge_model,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT,LLMTestCaseParams.EXPECTED_OUTPUT],
    evaluation_steps=[
        "Check if the actual answer provides the same information as expected",
        "Verify that 'Paris' is mentioned as the captial",
        "Ensure the anwser is factually correct"
    ],
    threshold=0.5,
    )

correctness_metric.measure(test_case)
print(f"Correctness Score: {correctness_metric.score}")
print(f"Reason:{correctness_metric.reason}")

if correctness_metric.is_successful():
    print("PASS")
else:
    print("FAIL")


bias_metric=BiasMetric(model=judge_model,threshold=0.5)

bias_metric.measure(test_case)
print(f"Bias metric Score: {bias_metric.score}")
print(f"Reason:{bias_metric.reason}")


tocxicity_metric=ToxicityMetric(model=judge_model,threshold=0.5)

tocxicity_metric.measure(test_case)
print(f"Toxicity metric Score: {tocxicity_metric.score}")
print(f"Reason:{tocxicity_metric.reason}")