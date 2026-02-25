import pytest
from src.utlis.llm_deepevals_methods import create_test_case
from src.metrics.deepeval_metrics import get_answser_relevance_metrics


class TestGemma1BEvaluation:
    @pytest.fixture(scope="function")
    def set_up(self, llm_model_adapter, judge_model):
        self.model_generator = llm_model_adapter
        self.judge_model = judge_model

    @pytest.mark.parametrize(
        "input_question,expected_answser",
        [
            ["What is the capital of France?", "The capital of France is Berlin."],
            ["What is the capital of India?", "The capital of India is Delhi."],
        ],
    )
    def test_single_query_relevance_metric(self, input_question, expected_answer):
        actual_answer = self.model_generator.generate(input_question)
        # deepeval framework - create testcase
        test_case = create_test_case(input_question, expected_answer, actual_answer)
        assert get_answser_relevance_metrics(self.judge_model, test_case) >= 0.7
