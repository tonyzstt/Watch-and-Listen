import json
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util


def compute_bleu_score(reference_summary, candidate_summary):

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


def load_metadata_for_video(metadata_folder, video_filename):
    """Loads metadata for a specific video from its corresponding JSON file."""
    metadata_file = os.path.join(metadata_folder, f"{os.path.splitext(video_filename)[0]}.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as file:
            metadata = json.load(file)
        return metadata["original_metadata"]["caption"]  # Extract the reference title
    return None

def load_generated_titles(output_file):
    """Loads generated titles from the output file."""
    generated_titles = {}
    with open(output_file, 'r') as file:
        for line in file:
            if ": " in line:  # Ensure correct format
                video_filename, generated_title = line.strip().split(".mp4: ", 1)
                generated_titles[video_filename] = generated_title
    return generated_titles

def evaluate_titles(metadata_file, output_file):
    """Compares generated titles with reference titles and computes BLEU and similarity scores."""
    generated_titles = load_generated_titles(output_file)

    results = {}

    for video_filename, generated_title in generated_titles.items():
        reference_title = load_metadata_for_video(metadata_folder, video_filename)

        if reference_title:
            bleu_score = compute_bleu_score(reference_title, generated_title)
            similarity_score = compute_similarity(reference_title, generated_title)

            results[video_filename] = {
                "reference_title": reference_title,
                "generated_title": generated_title,
                "bleu_score": bleu_score,
                "similarity_score": similarity_score
            }
        else:
            results[video_filename] = {
                "reference_title": None,
                "generated_title": generated_title,
                "bleu_score": None,
                "similarity_score": None,
                "error": "Metadata file not found"
            }
    return results


def save_results(results, output_json):
    """Saves evaluation results to a JSON file."""
    with open(output_json, 'w') as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    metadata_folder = "data/MMTrail/processed_data/test/metas"
    output_file = "benchmark/cogvlm2-llama3-caption/caption/output.json"  
    output_json = "results.json" 

    evaluation_results = evaluate_titles(metadata_folder, output_file)
    save_results(evaluation_results, output_json)

    print(f"Evaluation completed. Results saved to {output_json}")