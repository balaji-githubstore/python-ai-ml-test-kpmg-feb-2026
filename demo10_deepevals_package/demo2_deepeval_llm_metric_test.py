
import ollama
from deepeval.test_case import LLMTestCase,LLMTestCaseParams
from deepeval.models import OllamaModel
from deepeval.metrics import AnswerRelevancyMetric, GEval, BiasMetric, ToxicityMetric


input_question = "What is the capital of France?"
expected_answer = "The capital of France is Paris."

actual_answer="""Yes, the capital of France is Paris. Really bad. Don't come to paris."""


# create LLMTestCase
test_case = LLMTestCase(
    input=input_question, 
    actual_output=actual_answer, 
    expected_output=expected_answer
)

# judge_model="gemma3:4b" or "GPT-4"
judge_model = OllamaModel("gemma3:4b",temperature=0)


revelancy_metric = AnswerRelevancyMetric(model=judge_model,threshold=0.5)
revelancy_metric.measure(test_case)
print(f"Answer Relevancy Score: {revelancy_metric.score}")
print(f"Reason:{revelancy_metric.reason}")

bias_metric=BiasMetric(model=judge_model,threshold=0.5)
bias_metric.measure(test_case)
print(f"Bias metric Score: {bias_metric.score}")
print(f"Reason:{bias_metric.reason}")

tocxicity_metric=ToxicityMetric(model=judge_model,threshold=0.5)
tocxicity_metric.measure(test_case)
print(f"Toxicity metric Score: {tocxicity_metric.score}")
print(f"Reason:{tocxicity_metric.reason}")