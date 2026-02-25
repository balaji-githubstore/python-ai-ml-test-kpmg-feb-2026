
from src.models.ollama_adapter import OllamaAdapter
import pytest
from deepeval.models import OllamaModel

@pytest.fixture(scope="session")
def llm_model_adapter():
    return OllamaAdapter()

@pytest.fixture(scope="session")
def judge_model():
    return OllamaModel("gemma3:4b", temperature=0)