import argparse
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of model evaluation results by group.")
    parser.add_argument(
        "--input_excel", type=str, required=True, help="Input Excel file path (with multi-level columns)"
    )
    parser.add_argument("--output_txt", type=str, required=True, help="Output TXT file path")
    args = parser.parse_args()

    # Redirect all print output to both console and txt file
    class Tee(object):
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                try:
                    if not f.closed:
                        f.write(obj)
                        f.flush()
                except ValueError:
                    continue

        def flush(self):
            for f in self.files:
                try:
                    if not f.closed:
                        f.flush()
                except ValueError:
                    continue

    original_stdout = sys.stdout
    with open(args.output_txt, "w", encoding="utf-8") as ftxt:
        sys.stdout = Tee(sys.__stdout__, ftxt)
        try:
            # 1. Read data correctly (multi-level columns, model name as row index)
            df = pd.read_excel(args.input_excel, header=[0, 1], index_col=0)

            # 2. Group definitions
            basic_following = {
                "Attribute": ["shape+color", "color+texture", "texture+color", "shape+texture"],
                "Relation": ["2d_spatial_relation", "3d_spatial_relation", "action+2d", "action+3d"],
                "Reasoning": ["numeracy", "negation", "differentiation", "comparison"],
            }
            advanced_following = {
                "Attribute+Relation": [
                    "action+color",
                    "action+texture",
                    "color+2d",
                    "color+3d",
                    "shape+2d",
                    "shape+3d",
                    "texture+2d",
                    "texture+3d",
                ],
                "Attribute+Reasoning": [
                    "numeracy+color",
                    "numeracy+texture",
                    "comparison+color",
                    "comparison+texture",
                    "differentiation+color",
                    "differentiation+texture",
                    "negation+color",
                    "negation+texture",
                ],
                "Relation+Reasoning": [
                    "numeracy+2d",
                    "numeracy+3d",
                    "comparison+2d",
                    "comparison+3d",
                    "differentiation+2d",
                    "differentiation+3d",
                    "negation+2d",
                    "negation+3d",
                ],
                "Text Generation": ["text"],
                "Style Control": ["style"],
            }
            real_world_following = {"Complex": ["real_world"]}

            all_groups = {
                "Basic_Following_Overall": {"Overall": sum(basic_following.values(), [])},
                **{f"Basic_{k}": {k: v} for k, v in basic_following.items()},
                "Advanced_Following_Overall": {"Overall": sum(advanced_following.values(), [])},
                **{f"Advanced_{k}": {k: v} for k, v in advanced_following.items()},
                "Real_World_Following_Overall": {"Overall": sum(real_world_following.values(), [])},
                **{f"Real_{k}": {k: v} for k, v in real_world_following.items()},
            }

            # 3. Statistic function
            def calc_score_df(df, items):
                result = pd.DataFrame(index=df.index, columns=["short", "long"])
                for t in ["short", "long"]:
                    # Only count columns that exist
                    cols = [(attr, t) for attr in items if (attr, t) in df.columns]
                    if cols:
                        result[t] = df[cols].mean(axis=1)
                    else:
                        result[t] = float("nan")
                return result

            # 4. Output all group statistics
            for group_name, group_dict in all_groups.items():
                for subname, items in group_dict.items():
                    df_score = calc_score_df(df, items)
                    print(f"\n==== {group_name} - {subname} ====")
                    print(df_score.round(4))

            nine_groups = [
                ("Basic_Attribute", "Attribute"),
                ("Basic_Relation", "Relation"),
                ("Basic_Reasoning", "Reasoning"),
                ("Advanced_Attribute+Relation", "Attribute+Relation"),
                ("Advanced_Attribute+Reasoning", "Attribute+Reasoning"),
                ("Advanced_Relation+Reasoning", "Relation+Reasoning"),
                ("Advanced_Text Generation", "Text Generation"),
                ("Advanced_Style Control", "Style Control"),
                ("Real_Complex", "Complex"),
            ]

            # Collect scores for each sub-attribute
            model_scores = {}

            for group_name, subname in nine_groups:
                items = all_groups[group_name][subname]
                df_score = calc_score_df(df, items)
                # For each model, collect its short and long scores
                for model in df_score.index:
                    if model not in model_scores:
                        model_scores[model] = {"short": [], "long": []}
                    model_scores[model]["short"].append(df_score.loc[model, "short"])
                    model_scores[model]["long"].append(df_score.loc[model, "long"])

            # Calculate overall
            overall_rows = []
            for model, scores in model_scores.items():
                overall_short = pd.Series(scores["short"]).mean(skipna=True)
                overall_long = pd.Series(scores["long"]).mean(skipna=True)
                overall_rows.append([model, overall_short, overall_long])

            overall_df = pd.DataFrame(overall_rows, columns=["Model", "overall-short", "overall-long"]).set_index(
                "Model"
            )
            print("\n==== Overall average score of 9 sub-attributes ====")
            print(overall_df.round(4))
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
