import csv

major_class = []
minor_class = []
pattern_sum = []
rows = []  

# Read the CSV file
with open("ecoli.csv", 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    
    class_index = header.index('Class')
    
    for row in reader:
        if row[class_index] == '1':  # Major class
            major_class.append(row)
            features_sum = sum([float(value) for i, value in enumerate(row) if i != class_index])
            pattern_sum.append(features_sum)
            rows.append(row)
        elif row[class_index] == '0':  # Minor class
            minor_class.append(row)

# Combine rows with pattern sums
combined_rows = [(row, sum([float(value) for i, value in enumerate(row) if i != class_index])) for row in rows]

# Sort rows based on pattern_sum (increasing order)
combined_rows.sort(key=lambda x: x[1])

# Print sorted pattern sums
print("Pattern Sum in Sorted Order")
sorted_pattern_sum = [x[1] for x in combined_rows]
print(sorted_pattern_sum)

threshold = 0.4
new_cluster = []

# Check for differences and add to new_cluster
for i in range(len(combined_rows)):
    for j in range(i + 1, len(combined_rows)):  # Only compare each pair once
        difference = abs(combined_rows[i][1] - combined_rows[j][1])
        if difference > threshold:
            new_cluster.append(combined_rows[i][0])  # Add the row (not the pattern sum)

# Remove duplicates from new_cluster (set automatically removes duplicates)
new_cluster = list({tuple(row): row for row in new_cluster}.values())

# Print the new cluster without duplicates
print("New Cluster based on the difference from other values (no duplicates):")
for row in new_cluster:
    print(f"Row: {row}")

# Print the final length of new_cluster
print(f"Final length of the new_cluster list: {len(new_cluster)}")

# Combine rows and minor_class
combined_rows_final = new_cluster + minor_class

# Write the final combined dataset to a CSV file
op_file = 'final_data.csv'
with open(op_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)  # Write header
    writer.writerows(combined_rows_final)  # Write rows

print(f"Combined dataset has been written to '{op_file}'")
