import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV
path = "/Users/ellawiser/Desktop/DS-4420-final-project/Data/cancer_image_csv_data/breast_cancer_images.csv"
df = pd.read_csv(path)


# Optional: rename for nicer display
df["label"] = df["label"].replace({
    "normal": "Normal",
    "benign": "Benign"
})

# Custom colors
colors = ["#ffb6c1", "#ff1493"]   # light pink, dark pink

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="label", palette=colors)
plt.title("Image Count by Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()