"""
# pip install nltk rouge-score bert-score sentence-transformers pandas scikit-learn

import ollama
used to generate model output localy via ollama

import pandas as pd 
used to create and manipulate dataset using dataframe

import nltk
used for tokenization and BLEU score calculation 

rouge-score
compute rouge-score (ROUGE-1, ROUGE-L)

 sentence-transformer
 Used to load pre-trained model from hugging face 

"""

import ollama 
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from rouge_score import rouge_scorer

from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score

from sklearn.metrics.pairwise import cosine_similarity


"""Run only once. this will be downloaded -
C:\\Users\\admin\\AppData\\Roaming\\nltk_data"""
# nltk.download("punkt")
# nltk.download("punkt_tab")


data=[
    {
        "prompt":"Summarize: The cat sat on the mat.",
        "reference":"A cat was sitting on a mat."
    },
    {
         "prompt":"Explain machine learning.",
        "reference":"Machine learning is method where computer learns from data"
    }
]


df=pd.DataFrame(data)
print(df)

"""
BLEU/ROUGE 1 and ROUGE L  --> check words and not meaning 

BERTScore/SBERT --> does not check words but meaning 


BLEU --> runs ollama model gemma3:1b --> gets the ouput --> do ngram words check using output and reference
How many n-gram word sequence in the model ouput also apprears in reference?

ROUGE 1 --> Did the model use the same important words as the reference? 
measure --> unigram (single word overlap)


ROUGE L--> Did the model preserve the senstenace and word order? 
measure --> longest common subsequence (LCS)

BLEU: 
< 0.1 --> very low 
0.1 - 0.3 --> moderate 
0.4 --> strong lexical match 


ROUGE 1/ROUGE L
< 0.1 --> very low 
0.1 - 0.3 --> moderate 
0.4 --> strong overlap 

"""
# 


