import os
import re
from tabulate import tabulate

def parse_folder_name(folder_name):
    """Parse the folder name to extract num_shots, lambda_s, and lambda_q."""
    match = re.search(r"_(\d+)_ls(\d+(\.\d+)?)_lq(\d+(\.\d+)?)_", folder_name)
    if match:
        num_shots = match.group(1)
        lambda_s = match.group(2)
        lambda_q = match.group(4)
        return num_shots, lambda_s, lambda_q
    return None, None, None

def read_and_compute_mean_accuracy(file_path):
    """Read the file and compute the mean accuracy."""
    with open(file_path, 'r') as file:
        accuracies = []
        for line in file:
            match = re.search(r'acc: (\d+\.\d+)%', line)
            if match:
                accuracies.append(float(match.group(1)))
        return sum(accuracies) / len(accuracies) if accuracies else None

def create_dataset_dictionaries(root_dataset_directory):
    """Create nested dictionaries for each dataset with mean accuracy."""
    dataset_dictionaries = {}

    # List all dataset folders in the root directory
    for dataset_folder in os.listdir(root_dataset_directory):
        dataset_path = os.path.join(root_dataset_directory, dataset_folder)

        # Check if it's a directory
        if os.path.isdir(dataset_path):
            num_shots_dict = {}
            # Iterate through subfolders in the dataset folder
            for subfolder in os.listdir(dataset_path):
                num_shots, lambda_s, lambda_q = parse_folder_name(subfolder)

                # Skip if lambda_q is 1.0 or if parsing failed
                if lambda_q == '1.0' or num_shots is None:
                    continue

                # Initialize num_shots dict if not present
                if num_shots not in num_shots_dict:
                    num_shots_dict[num_shots] = {}

                # Path to the accuracy file
                accuracy_file_path = os.path.join(dataset_path, subfolder, 'per_class_results_test_0.txt')

                # Compute mean accuracy
                mean_accuracy = read_and_compute_mean_accuracy(accuracy_file_path) if os.path.exists(accuracy_file_path) else None

                # Add lambda_s and mean accuracy to the num_shots dict
                num_shots_dict[num_shots][lambda_s] = {'subfolder': subfolder, 'mean_accuracy': mean_accuracy}

            dataset_dictionaries[dataset_folder] = num_shots_dict

    return dataset_dictionaries


def print_tables(dataset_dictionaries):
    """Print tables for each num_shots with dataset names and lambda_s accuracies."""
    num_shots_values = sorted({int(key) for d in dataset_dictionaries.values() for key in d.keys()})

    for num_shots in num_shots_values:
        print(f"Number of Shots: {num_shots}")

        # Collecting table data
        table_data = []
        lambda_s_values = sorted(
            {lambda_s for d in dataset_dictionaries.values() for lambda_s in d.get(str(num_shots), {})})
        headers = ["Dataset"] + lambda_s_values

        for dataset, num_shots_dict in dataset_dictionaries.items():
            row = [dataset]
            for lambda_s in lambda_s_values:
                accuracy = num_shots_dict.get(str(num_shots), {}).get(lambda_s, {}).get("mean_accuracy")
                row.append(accuracy if accuracy is not None else "N/A")
            table_data.append(row)

        # Print the table
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()


# Example usage
root_dataset_directory = '/home/gerinb/logs/_search'  # Replace with your actual path
dictionaries = create_dataset_dictionaries(root_dataset_directory)
print(dictionaries)

print(len(dictionaries.keys()))

print(len(dictionaries['SSEuroSAT'].keys()))
print(len(dictionaries['SSDescribableTextures'].keys()))
print_tables(dictionaries)