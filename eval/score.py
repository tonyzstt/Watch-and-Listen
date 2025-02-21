import json

def compute_overall_scores(evaluation_results_file, output_summary_file):
    """Computes the average BLEU score and similarity score from the evaluation results."""
    
    with open(evaluation_results_file, 'r') as file:
        results = json.load(file)

    total_bleu = 0
    total_similarity = 0
    count = 0

    for video, scores in results.items():
        bleu_score = scores.get("bleu_score")
        similarity_score = scores.get("similarity_score")

        if bleu_score is not None and similarity_score is not None:
            total_bleu += bleu_score
            total_similarity += similarity_score
            count += 1

    # Compute averages, avoiding division by zero
    avg_bleu = total_bleu / count if count > 0 else 0
    avg_similarity = total_similarity / count if count > 0 else 0

    summary = {
        "average_bleu_score": avg_bleu,
        "average_similarity_score": avg_similarity,
        "total_evaluated": count
    }

    # Save summary results
    with open(output_summary_file, 'w') as file:
        json.dump(summary, file, indent=4)

    print(f"Overall scores saved to {output_summary_file}")
    return summary

if __name__ == "__main__":
    evaluation_results_file = "results.json"  # Replace with your actual path
    output_summary_file = "scores.json"  # Replace with your desired output file

    overall_scores = compute_overall_scores(evaluation_results_file, output_summary_file)
    print(overall_scores)
