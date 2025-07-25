import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Get current script directory
base_path = Path(__file__).parent  # Directory of this script

# File name (Excel file must be in the same folder)
file_name = "data_rule_of_40.xlsx"
file_path = base_path / file_name

# Read Excel file
df = pd.read_excel(file_path)

# Calculate Rule of 40
df["Rule of 40"] = df["Revenue Growth %"] + df["Profit Margin %"]

# Display DataFrame
print(df)

# Plot line chart for Revenue Growth, Profit Margin, and Rule of 40
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["Revenue Growth %"], marker='o', label="Revenue Growth %")
plt.plot(df["Year"], df["Profit Margin %"], marker='o', label="Profit Margin %")
plt.plot(df["Year"], df["Rule of 40"], marker='o', linewidth=2.5, label="Rule of 40", color='black')

# Add threshold line for Rule of 40
plt.axhline(y=40, color='blue', linestyle='--', linewidth=2, label="Rule of 40 Threshold")

# Annotate Rule of 40 values on the chart
for i, value in enumerate(df["Rule of 40"]):
    plt.text(df["Year"].iloc[i], value + 1 , f"{value}%", ha='center', fontsize=11, color='black')

# Titles and labels
plt.title("Rule of 40 Analysis Over Time (Company XYZ)")
plt.xlabel("Year")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()