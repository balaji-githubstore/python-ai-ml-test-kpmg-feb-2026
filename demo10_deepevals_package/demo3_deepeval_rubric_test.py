import ollama
from deepeval.test_case import LLMTestCase,LLMTestCaseParams
from deepeval.models import OllamaModel
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric


input_question = "What is the capital of France?"
expected_answer = "The capital of France is Paris."

actual_answer="""Yes, the capital of France is Berlin. It'a nice place"""


# create LLMTestCase
test_case = LLMTestCase(
    input=input_question, 
    actual_output=actual_answer, 
    expected_output=expected_answer
)

judge_model = OllamaModel("gemma3:4b",temperature=0)

correctness_metric=GEval(
    name="Correctness",
    model=judge_model,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT,LLMTestCaseParams.EXPECTED_OUTPUT],
    rubric=[
        Rubric(score_range=(0,2),expected_outcome="Factually Incorrect"),
        Rubric(score_range=(3,6),expected_outcome="Mostly Correct"),
        Rubric(score_range=(7,9),expected_outcome="Correct but missing minor details"),
        Rubric(score_range=(10,10),expected_outcome="100% Correct")
    ],
    evaluation_steps=[
        "Check if the actual answer provides the same information as expected",
        "Check for contradictions",
        "Ignore harmless extra details",
        "Assign score based on the rubric"
    ],
    threshold=0.3,
    )

correctness_metric.measure(test_case)
print(f"Correctness Score: {correctness_metric.score}")
print(f"Reason:{correctness_metric.reason}")

if correctness_metric.is_successful():
    print("PASS")
else:
    print("FAIL")