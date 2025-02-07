import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util


def compute_blue_score(reference_summary, candidate_summary):

    reference_tokens = nltk.word_tokenize(reference_summary.lower())
    candidate_tokens = nltk.word_tokenize(candidate_summary.lower())

    references = [reference_tokens]
    candidate = candidate_tokens

    chencherry = SmoothingFunction()

    bleu_score = sentence_bleu(
        references,
        candidate,
        smoothing_function=chencherry.method1
    )

    return bleu_score

def compute_similarity(reference_summary, candidate_summary):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    embedding1 = model.encode(reference_summary, convert_to_tensor=True)
    embedding2 = model.encode(candidate_summary, convert_to_tensor=True)
    similarity = util.cos_sim(embedding1, embedding2)
    return similarity.item()

