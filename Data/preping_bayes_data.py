import pandas as pd

path = "/Users/ellawiser/Desktop/DS-4420-final-project/Data/breast-cancer-wisconsin.data"

# column names
columns = [
    "sample_code",
    "clump_thickness",
    "uniformity_cell_size",
    "uniformity_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "class"
]

# load data
df = pd.read_csv(
    path,
    header=None,
    names=columns,
    na_values="?"
)

# drop rows with missing values (bare nuclei often has "?")
df = df.dropna()

# Convert numeric columns to integer
numeric_cols = columns[:-1]
df[numeric_cols] = df[numeric_cols].astype(int)

# Convert class values: 2 -> B benign, 4 -> M malignant
df["class"] = df["class"].map({2: "B", 4: "M"})

# save cleaned CSV
out_path = "/Users/ellawiser/Desktop/DS-4420-final-project/Data/bayes_breast_cancer_clean.csv"
df.to_csv(out_path, index=False)

print("Saved cleaned CSV to:", out_path)
print(df.head())