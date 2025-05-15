import json
import matplotlib.pyplot as plt
from collections import Counter

# Function to extract case names, links, and jurisdiction from the first 110 lines of a JSONL file
def extract_case_details(jsonl_file):
    case_details = []
    jurisdictions = []  # List to store jurisdictions for the pie chart
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        # Read and process the first 110 lines
        for i, line in enumerate(f):
            if i >= 200:  # Stop after 110 lines
                break
            try:
                case = json.loads(line)
                case_name = case.get("medium_neutral_citation", "N/A")
                case_link = case.get("url", "N/A")
                jurisdiction = case.get("jurisdiction", "N/A")
                
                # Collect case details
                case_details.append({
                    "case_name": case_name,
                    "case_link": case_link,
                    "jurisdiction": jurisdiction
                })
                
                # Add the jurisdiction to the list for pie chart
                jurisdictions.append(jurisdiction)
                
            except json.JSONDecodeError:
                print(f"Skipping invalid line {i+1}")
    
    return case_details, jurisdictions

# Replace 'your_file.jsonl' with the path to your JSONL file
jsonl_file = r'C:\Users\hp\lawRAG2\data\cases_2.jsonl'
case_details, jurisdictions = extract_case_details(jsonl_file)

# Create a pie chart for jurisdiction distribution
jurisdiction_counts = Counter(jurisdictions)

# Group small categories into "Others" if they make up less than a threshold (e.g., 5%)
threshold = 0.03  # 5% of total cases
total_cases = sum(jurisdiction_counts.values())

# Create a dictionary to hold the grouped jurisdictions
grouped_jurisdictions = {}
others_count = 0

# Iterate through the counts and group small categories
for jurisdiction, count in jurisdiction_counts.items():
    if count / total_cases < threshold:
        others_count += count
    else:
        grouped_jurisdictions[jurisdiction] = count

# Add the "Others" category
if others_count > 0:
    grouped_jurisdictions["Others"] = others_count

# Prepare the data for the pie chart
labels = list(grouped_jurisdictions.keys())
sizes = list(grouped_jurisdictions.values())
explode = [0.1] * len(labels)  # Slightly explode each slice for visibility

# Plot the pie chart
plt.figure(figsize=(10, 10))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, explode=explode)

# Add a legend for clarity
plt.legend(labels, loc='best', fontsize=8)

# Set chart title and show the plot
plt.title("Case Jurisdictions Distribution")
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
plt.show()
