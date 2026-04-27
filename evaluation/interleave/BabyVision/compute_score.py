import json
import argparse
import sys

type_match = {
    "Fine-grained Discrimination": "1",
    "Visual Tracking": "2",
    "Spatial Perception": "3",
    "Visual Pattern Recognition": "4",
}


def compute_avg_subscores_for_Type_and_Subtype(path):
    # compute 3 scores, 1) overall accuracy 2) Type average accuracy 3) Subtype average accuracy
    with open(path, "r") as f:
        results = json.load(f)
    total = len(results)
    correct = sum(1 for r in results if r['LLMJudgeResult'])
    overall_accuracy = correct / total if total > 0 else 0.0
#    print(f"Overall Accuracy: {overall_accuracy:.4f} ({correct}/{total})")
    
    type_dict = {}
    subtype_dict = {}
    
    for r in results:
        t = r['Type']
        st = r['Subtype']
        
        type_id = type_match.get(t, "0")
        
        if t not in type_dict:
            type_dict[t] = {'total': 0, 'correct': 0}
        if type_id+t+"/"+st not in subtype_dict:
            subtype_dict[type_id+t+"/"+st] = {'total': 0, 'correct': 0}

        type_dict[t]['total'] += 1
        subtype_dict[type_id+t+"/"+st]['total'] += 1
        if r['LLMJudgeResult']:
            type_dict[t]['correct'] += 1
            subtype_dict[type_id+t+"/"+st]['correct'] += 1
        
    return overall_accuracy, type_dict, subtype_dict

def compute_average_and_std(results_list):
    import numpy as np
    overall_accuracies = [r[0] for r in results_list]
    type_dicts = [r[1] for r in results_list]
    subtype_dicts = [r[2] for r in results_list]
    
    overall_avg = np.mean(overall_accuracies)
    overall_std = np.std(overall_accuracies)
    
    print(f"\nOverall Average Accuracy: {overall_avg:.4f} ± {overall_std:.4f}")
    
    # Type-wise
    all_types = set()
    for td in type_dicts:
        all_types.update(td.keys())
    
    print("\nType-wise Average Accuracy:")
    for t in sorted(all_types):
        accs = []
        for td in type_dicts:
            if t in td:
                acc = td[t]['correct'] / td[t]['total'] if td[t]['total'] > 0 else 0.0
                accs.append(acc)
        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"  {t}: {avg_acc:.4f} ± {std_acc:.4f}")
    
    # Subtype-wise
    all_subtypes = set()
    for std in subtype_dicts:
        all_subtypes.update(std.keys())
    
    subtypes_acc = {}
    
    
    print("\nSubtype-wise Average Accuracy:")
    for st in all_subtypes:
        accs = []
        for std in subtype_dicts:
            if st in std:
                acc = std[st]['correct'] / std[st]['total'] if std[st]['total'] > 0 else 0.0
                accs.append(acc)
        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        
        subtypes_acc[st] = (avg_acc, std_acc)

    
    for st, (avg_acc, std_acc) in sorted(subtypes_acc.items(), key=lambda x: x[0]):
        print(f"  {st}: {avg_acc:.4f} ± {std_acc:.4f}")

if __name__=="__main__":
    # read the pass file from CLI and compute the scores
    parser = argparse.ArgumentParser(description='Compute average scores from multiple result JSON files')
    parser.add_argument('files', nargs='+', help='Path(s) to result JSON file(s)')
    
    args = parser.parse_args()
    
    if len(args.files) == 0:
        print("Error: At least one file path is required", file=sys.stderr)
        sys.exit(1)
    
    results_list = []
    for file_path in args.files:
        try:
            result = compute_avg_subscores_for_Type_and_Subtype(file_path)
            results_list.append(result)
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file: {file_path}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)
            sys.exit(1)
    
    if len(results_list) > 0:
        compute_average_and_std(results_list)
    else:
        print("Error: No valid results to process", file=sys.stderr)
        sys.exit(1)