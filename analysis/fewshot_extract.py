import os
import re
from collections import defaultdict


def parse_filename(filename):
    """Parse filename to extract architecture, dataset, n_shots, and seed."""
    match = re.match(r'(.+)_(.+)_(\d+)shots_seed(\d+)\.txt', filename)
    if match:
        return match.groups()
    return None


def extract_accuracies_from_file(filepath):
    """Extract the highest validation accuracy and its subsequent test accuracy."""
    with open(filepath, 'r') as file:
        lines = file.readlines()

    max_val_acc = 0
    test_acc_for_max_val = None
    for i, line in enumerate(lines):
        if "Validation accuracy:" in line:
            val_acc = float(line.split(":")[-1].strip())
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                # Ensure the next line contains the test accuracy
                if i - 1 < len(lines) and "* average:" in lines[i - 1]:
                    test_acc_for_max_val = round(float(lines[i-1].split(":")[-1].strip().rstrip('% \n')), 2)


    return test_acc_for_max_val


def compute_mean_accuracies(folder_path):
    """Compute mean accuracies for each arch, dataset, and n_shots across seeds."""
    accuracies = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            parsed = parse_filename(filename)
            if parsed:
                arch, dataset, n_shots, seed = parsed
                filepath = os.path.join(folder_path, filename)
                test_acc = extract_accuracies_from_file(filepath)
                if test_acc is not None:
                    accuracies[arch][dataset][n_shots].append(test_acc)

    # Compute mean accuracies
    mean_accuracies = defaultdict(dict)
    for arch, datasets in accuracies.items():
        for dataset, n_shots_data in datasets.items():
            for n_shots, test_accs in n_shots_data.items():
                mean_acc = sum(test_accs) / len(test_accs)
                mean_accuracies[(arch, dataset, n_shots)] = round(mean_acc, 2)

    return mean_accuracies


output_file_path = 'fewshot_mean_accuracies_summary.txt'

# Usage
folder_path = '/export/home/gerinb/trainval_logs'
mean_accuracies = compute_mean_accuracies(folder_path)

# Sort mean_accuracies by arch, dataset, and n_shots for printing
sorted_keys = sorted(mean_accuracies.keys(), key=lambda x: (x[0], x[1], int(x[2])))

# Open the file for writing
with open(output_file_path, 'w') as file:
    for key in sorted_keys:
        arch, dataset, n_shots = key
        mean_acc = mean_accuracies[key]
        # Write to the file instead of printing
        file.write(f"{arch}, {dataset}, {n_shots} shots: Mean Test Accuracy = {mean_acc}%\n")

print(f"Results have been saved to {output_file_path}.")

