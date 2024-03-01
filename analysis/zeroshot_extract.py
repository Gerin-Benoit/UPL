import os
import re
from collections import defaultdict

# Adjust these paths as necessary
folder_path = '/gpfs/home/acad/ucl-elen/gerinb/train_logs_TEMPLATES'
output_file_path = 'CT_zeroshot_mean_accuracies.txt'

# Regular expression to match the lines with accuracies and to parse filenames
accuracy_pattern = re.compile(r'\* average: (\d+\.\d+)%')
filename_pattern = re.compile(r'anay_(.*?)_(.*?).txt')

# Store mean accuracies and accumulative accuracies for computing mean of means
mean_accuracies = defaultdict(list)
arch_accuracies = defaultdict(list)

for filename in os.listdir(folder_path):
    match = filename_pattern.match(filename)
    if match:
        arch, dataset = match.groups()
        #if dataset.startswith("ssimagenet"):
        #    continue
        filepath = os.path.join(folder_path, filename)
        accuracies = []

        with open(filepath, 'r') as file:
            for line in file:
                accuracy_match = accuracy_pattern.search(line)
                if accuracy_match:
                    accuracies.append(float(accuracy_match.group(1)))

        if accuracies:
            mean_accuracy = sum(accuracies) / len(accuracies)
            mean_accuracies[arch].append((dataset, round(mean_accuracy, 2)))
            arch_accuracies[arch].append(mean_accuracy)

# Sort the results by arch and then dataset, and calculate mean of means
sorted_archs = sorted(mean_accuracies.keys())
mean_of_means = {arch: round(sum(arch_accuracies[arch]) / len(arch_accuracies[arch]), 2) for arch in sorted_archs}

# Save the sorted results to a file and print mean of means
with open(output_file_path, 'w') as file:
    for arch in sorted_archs:
        file.write(f"Architecture: {arch}\n")
        for dataset, mean_accuracy in sorted(mean_accuracies[arch], key=lambda x: x[0]):
            file.write(f"  {dataset}: {mean_accuracy}%\n")
        file.write(f"  Mean of Mean Accuracies for {arch}: {mean_of_means[arch]}%\n\n")

print(f"Mean accuracies and Mean of Mean accuracies saved to {output_file_path}.")

