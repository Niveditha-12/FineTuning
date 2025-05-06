import json

# Step 1: Load full dataset
with open("alpaca_data.json", "r") as f:
    full_data = json.load(f)

# Step 2: Extract first 200 examples
subset = full_data[:200]

# Step 3: Save subset to new file
with open("data/alpaca_200.json", "w") as f:
    json.dump(subset, f, indent=2)

print("Saved 200-sample Alpaca dataset to data/alpaca_200.json ✅")
