import pandas as pd
# Load datasets
real_df = pd.read_excel("Real.xlsx")
fake_df = pd.read_excel("Fake.xlsx")

# Keep only English rows
real_english = real_df[real_df["language"].str.lower() == "english"]
fake_english = fake_df[fake_df["language"].str.lower() == "english"]

# Randomly sample 400 rows from each
real_sample = real_english.sample(n=400, random_state=42)
fake_sample = fake_english.sample(n=400, random_state=42)

# Combine both datasets
combined_df = pd.concat([real_sample, fake_sample], ignore_index=True)

# Shuffle rows randomly so labels are mixed naturally
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save final dataset
combined_df.to_excel("Combined_800_English.xlsx", index=False)

print("Combined dataset created successfully!")
print(combined_df["label"].value_counts())


