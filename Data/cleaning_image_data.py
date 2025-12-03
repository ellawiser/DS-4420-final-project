import os
import pandas as pd

# Dataset root
ROOT_DIR = "1/Dataset_BUSI_with_GT"

# Create a list of classes
classes = ["benign", "malignant", "normal"]

rows = []
index = 1

for cls in classes:
    folder = os.path.join(ROOT_DIR, cls)
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".png"):
            continue

        # skip mask images like "benign (1)_mask.png"
        if "_mask" in fname:
            continue
        # skip mask images like "benign (1)_mask.png"
        if "malignant" in fname:
                continue

        img_path = os.path.join(folder, fname)

        rows.append({
            "index": index,
            "image_name": fname,
            "image_path": img_path,
            "classification": cls
        })

        index += 1

# build df
df = pd.DataFrame(rows)

# make new folder
OUTPUT_DIR = "cancer_image_csv_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# save csv file
output_csv = os.path.join(OUTPUT_DIR, "breast_cancer_images.csv")
df.to_csv(output_csv, index=False)

print("Saved new file to:", output_csv)
print(df.head())