import json

def compute_overall_scores(evaluation_results_file, output_summary_file):
    """Computes the average BLEU score, similarity score, precision, recall, and F1-score from the evaluation results."""
    
    with open(evaluation_results_file, 'r') as file:
        results = json.load(file)

    total_bleu = 0
    total_similarity = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    count = 0

    for video, scores in results.items():
        bleu_score = scores.get("bleu_score")
        similarity_score = scores.get("similarity_score")
        precision = scores.get("precision")
        recall = scores.get("recall")
        f1_score = scores.get("f1")  # 直接从数据中获取 F1-score

        if None not in (bleu_score, similarity_score, precision, recall, f1_score):
            total_bleu += bleu_score
            total_similarity += similarity_score
            total_precision += precision
            total_recall += recall
            total_f1 += f1_score
            count += 1

    # Compute averages, avoiding division by zero
    avg_bleu = total_bleu / count if count > 0 else 0
    avg_similarity = total_similarity / count if count > 0 else 0
    avg_precision = total_precision / count if count > 0 else 0
    avg_recall = total_recall / count if count > 0 else 0
    avg_f1 = total_f1 / count if count > 0 else 0  # 直接计算平均 F1-score

    summary = {
        "average_bleu_score": avg_bleu,
        "average_similarity_score": avg_similarity,
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1_score": avg_f1,
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
