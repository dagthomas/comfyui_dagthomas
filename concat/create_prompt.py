import json
import random

# Step 1: Load JSON data from a file with UTF-8 encoding
with open('output.json', 'r', encoding='utf-8') as file:
    json_list = json.load(file)

# Step 2: Randomly select 5 elements from the list
random_values = random.sample(json_list, 10)

# Step 3: Join the selected elements into a single string separated by commas
result_string = ", ".join(random_values)

# Step 4: Output the result
print(result_string)