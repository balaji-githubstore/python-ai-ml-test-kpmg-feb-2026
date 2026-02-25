from deepeval.test_case import LLMTestCase

def create_test_case(input_question,actual_answer,expected_answer,context=None,retrieved_context=None):
    return LLMTestCase(
            input=input_question,
            actual_output=actual_answer,
            expected_output=expected_answer,
            context=context,
            retrieval_context=retrieved_context
        )