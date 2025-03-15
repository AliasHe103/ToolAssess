import json

# Load dataset
filepath = "../data/single_task.json"
with open(filepath, 'r', encoding="utf-8") as file:
    data = json.load(file)

# Count occurrences of each category
results_count = {"requires tool": 0, "cannot be completed": 0, "no tool": 0}

for op in data.values():
    result = op["result"]
    results_count[result] += 1

# Calculate proportions
total = sum(results_count.values())
proportions = {key: f"{(value / total * 100):.2f}%" for key, value in results_count.items()}

# Display results
print("Results：")
for category, proportion in proportions.items():
    print(f"{category}: {proportion}")
