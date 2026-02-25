from deepeval.metrics import AnswerRelevancyMetric

def get_answser_relevance_metrics(judge_model,test_case):
    revelancy_metric = AnswerRelevancyMetric(model=judge_model, threshold=0.5)
    revelancy_metric.measure(test_case)
    return revelancy_metric.score