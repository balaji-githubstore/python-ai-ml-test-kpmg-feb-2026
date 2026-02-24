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
        "prompt":"Summarize: The cat is sitting on the mat.",
        "reference":"A dog is running in the park."
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

BERTScore --> does not check only words but meaning of the word. 
/SBERT --> does not check words but meaning of the sentence.

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

BERTScore F1:
>0.90 --> very high semantic similarity
0.85 - 0.90 --> Good
0.80 - 0.85 --> ok
<0.80 --> weak 

SBERT cosine similarity:
>0.80 --> Excellent
070 - 0.80 --> Good
0.60 - 0.70 --> OK
<0.60 --> weak


"""
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# hugging face model for sentence embedding. it will convert the sentence into vector representation.
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# To avoid zero BLEU score for short sentences, we can use smoothing techniques.
smooth=SmoothingFunction().method1


results = []

for index, row in df.iterrows():
    prompt = row['prompt']
    reference=row['reference']

    response=ollama.chat(model="gemma3:1b",messages=[{"role": "user", "content": prompt}])
    prediction=response.message.content

    ref_token=nltk.word_tokenize(reference)
    pred_token=nltk.word_tokenize(prediction)

    bleu_score=sentence_bleu([ref_token], pred_token, smoothing_function=smooth)

    rouge_score=rouge.score(reference, prediction)
    
    rouge_f1=rouge_score['rouge1'].fmeasure
    rouge_l=rouge_score['rougeL'].fmeasure

    """
    BERTScore --> evaluates the meaning preservation between the prediction and reference at work (token) level.
    P-> how many predicted positives are actully positive.  
    R-> how many actual positives were identitifed. 
    F1-> harmonic mean of precision and recall. final semantic similarity score. 
    
    """

    P,R,F1=bert_score([prediction], [reference], lang='en', verbose=False)

    bert_f1=F1.item()

    """
    SBert --> evaluates the meaning preservation between the prediction and reference at sentence level.
    1-> identical meaning
    0-> unrelated meaning 
    -1 -> opposite meaning
    
    """

    embeddings=sbert_model.encode([prediction, reference])
    cosine_sim=cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    results.append({
        # "prompt":prompt,
        # "reference":reference,
        # "prediction":prediction,
        "bleu":round(bleu_score, 4),
        "rouge1":round(rouge_f1, 4),
        "rougeL":round(rouge_l, 4),
        "bert_f1":round(bert_f1, 4),
        "sbert_cosine":round(cosine_sim, 4)
    })

    print(f"Prompt: {prompt}")
    print(f"Reference: {reference}")
    print(f"Prediction: {prediction}")



results_df=pd.DataFrame(results)
print(results_df)