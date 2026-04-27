import argparse
import json
import os
from collections import defaultdict

import pandas as pd

DEFAULT_INPUT_JSON = None
DEFAULT_OUTPUT_CSV = os.path.join("Interleaved_Arena", "model_score_summaries.csv")
DEFAULT_FILTERED_JSON = os.path.join("Interleaved_Arena", "gpt-score_results.json")

EXPECTED_SCORE_METRICS = [
    "Correctness",
    "Image-Text Coherency",
    "Multi-step Consistency",
    "Content Quality",
    "Human Preference Alignment",
    "Completeness",
    "Content Richness",
]


def normalize_judge_score(score):
    if not isinstance(score, dict):
        return score
    wrapped_scores = score.get("scores")
    if isinstance(wrapped_scores, dict):
        return wrapped_scores
    return score


def process_gpt_scores(
    input_json_path=DEFAULT_INPUT_JSON, csv_path=DEFAULT_OUTPUT_CSV, filtered_json_path=DEFAULT_FILTERED_JSON
):
    if not input_json_path:
        raise ValueError("input_json_path is required")
    # Read the JSON file
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Count original lines (entries)
    original_count = len(data)
    print(f"Original number of entries: {original_count}")

    # Filter out entries where score is null or empty
    filtered_data = []
    model_scores = defaultdict(lambda: defaultdict(list))  # model_id -> metric -> scores
    model_names = {}  # Store model names for each model_id
    invalid_score_type_count = 0

    for entry in data:
        score = normalize_judge_score(entry.get("score", {}))
        model = entry.get("model", {})
        model_id = model.get("id", "")
        model_name = model.get("name", "")

        # Store model name for this model_id
        model_names[model_id] = model_name

        # Check if score is null, empty, or contains null scores
        if score is None or score == {}:
            continue

        if not isinstance(score, dict):
            invalid_score_type_count += 1
            continue

        # Check if required metrics exist and any metric_data['Score'] is None
        has_null_score = False
        for metric in EXPECTED_SCORE_METRICS:
            metric_data = score.get(metric)
            if not isinstance(metric_data, dict) or "Score" not in metric_data:
                has_null_score = True
                break
            if metric_data["Score"] is None:
                has_null_score = True
                break

        # If any null score is found, skip this entry
        if has_null_score:
            continue

        # If we reach here, the entry has valid scores
        filtered_data.append(entry)

        # Collect scores for each model and metric
        for metric, metric_data in score.items():
            if isinstance(metric_data, dict) and "Score" in metric_data:
                # Only add non-None scores
                if metric_data["Score"] is not None:
                    model_scores[model_id][metric].append(metric_data["Score"])

    # Count filtered entries
    filtered_count = len(filtered_data)
    print(f"Number of entries after filtering: {filtered_count}")
    print(f"Removed {original_count - filtered_count} entries with null scores")
    if invalid_score_type_count:
        print(f"Skipped {invalid_score_type_count} entries with non-dict score values")

    # Calculate average scores for each model and metric
    model_summaries = {}
    for model_id, metrics in model_scores.items():
        model_summary = {}
        for metric, scores in metrics.items():
            if scores:  # Only calculate if there are scores
                # Ensure all scores are numeric types
                valid_scores = [s for s in scores if s is not None and isinstance(s, (int, float))]
                if valid_scores:
                    avg_score = sum(valid_scores) / len(valid_scores)
                    model_summary[metric] = round(avg_score, 2)

        # Calculate overall average
        all_scores = []
        for metric_avg in model_summary.values():
            all_scores.append(metric_avg)

        if all_scores:
            overall_avg = sum(all_scores) / len(all_scores)
            model_summary["Overall"] = round(overall_avg, 2)

        model_summaries[model_id] = model_summary

    # Print results
    print("\n=== Model Score Summaries ===")
    for model_id, summary in model_summaries.items():
        model_name = model_names.get(model_id, "Unknown")
        print(f"\nModel ID: {model_id} ({model_name})")
        for metric, avg_score in summary.items():
            print(f"  {metric}: {avg_score}")

    # Create DataFrame for CSV export with ranking
    csv_data = []
    for model_id, summary in model_summaries.items():
        model_name = model_names.get(model_id, "Unknown")
        row = {"Model_ID": model_id, "Model_Name": model_name}
        for metric, avg_score in summary.items():
            row[f"{metric}"] = avg_score
        csv_data.append(row)

    # Convert to DataFrame, sort by Overall score (descending), and add rank
    df = pd.DataFrame(csv_data)
    if df.empty:
        print("No valid scores found after filtering.")
        return
    df = df.sort_values("Overall", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nResults saved to: {csv_path}")

    # Write back filtered data to the original file
    with open(filtered_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print("File updated successfully!")
    return df


def plot_model_rankings(csv_path=DEFAULT_OUTPUT_CSV, show_plot=False):
    """Plot model ranking table"""
    import matplotlib.pyplot as plt

    # Read CSV data
    df = pd.read_csv(csv_path)

    # Set Chinese font
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # Create table data - put Overall first
    metrics = [
        "Overall",
        "Completeness",
        "Content Quality",
        "Content Richness",
        "Correctness",
        "Human Preference\nAlignment",
        "Image-Text\nCoherency",
        "Multi-step\nConsistency",
    ]

    # Prepare table data
    table_data = []
    for _, row in df.iterrows():
        table_row = [f"{row['Rank']}. {row['Model_Name']}"]
        for metric in metrics:
            # Handle metric names with line breaks
            metric_key = metric.replace("\n", " ")
            table_row.append(f"{row[metric_key]:.2f}")
        table_data.append(table_row)
    # Create table column headers
    column_labels = ["Model"] + metrics

    # Create chart
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis("tight")
    ax.axis("off")

    # Create table
    table = ax.table(cellText=table_data, colLabels=column_labels, cellLoc="center", loc="center", bbox=[0, 0, 1, 1])

    # Set table style - increase font size
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 2.2)

    # Set header style - use light color scheme
    for i in range(len(column_labels)):
        table[(0, i)].set_facecolor("#E3F2FD")  # Light blue
        table[(0, i)].set_text_props(weight="bold", color="#1976D2")

    # Set ranking style
    for i in range(len(table_data)):
        table[(i + 1, 0)].set_facecolor("#F3E5F5")  # Light purple
        table[(i + 1, 0)].set_text_props(weight="bold", color="#7B1FA2")

    # Set data cell style - use more harmonious light color scheme
    for i in range(1, len(table_data) + 1):
        for j in range(1, len(column_labels)):
            cell = table[(i, j)]
            value = float(table_data[i - 1][j])
            if value >= 8.0:
                cell.set_facecolor("#E8F5E8")  # Light green
            elif value >= 6.0:
                cell.set_facecolor("#FFF8E1")  # Light yellow
            elif value >= 4.0:
                cell.set_facecolor("#FFEBEE")  # Light pink
            else:
                cell.set_facecolor("#FCE4EC")  # Lighter pink

    plt.title(
        "Model Leaderboard Ranked by GPT-based Subjective Scores",
        fontsize=18,
        fontweight="bold",
        pad=25,
        color="#424242",
    )
    plt.tight_layout()

    # plt.savefig('Interleaved_Arena/model_rankings_table.png', dpi=300, bbox_inches='tight')
    # print("Table saved to: Interleaved_Arena/model_rankings_table.png")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize GPT scoring results")
    parser.add_argument("--input_json", type=str, required=True, help="Input GPT score JSON file")
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Output summary CSV path")
    parser.add_argument("--filtered_json", type=str, default=DEFAULT_FILTERED_JSON, help="Output filtered JSON path")
    parser.add_argument("--plot", action="store_true", help="Show the ranking table plot")
    args = parser.parse_args()

    process_gpt_scores(
        input_json_path=args.input_json,
        csv_path=args.output_csv,
        filtered_json_path=args.filtered_json,
    )
    if args.plot:
        plot_model_rankings(csv_path=args.output_csv, show_plot=args.plot)
