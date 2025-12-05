"""
cluster_legal_texts_casewise.py
---------------------------------
Clusters each legal PDF separately and generates a PCA plot per case.
Each cluster is labeled with its top context words (not just 'Cluster 0').
"""

import os
import re
import pdfplumber
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============================
# 📁 PATH SETUP
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"[INFO] Data directory: {DATA_DIR}")
print(f"[INFO] Results will be saved to: {RESULTS_DIR}")


# ============================
# 🧹 TEXT PROCESSING
# ============================
def extract_pdf_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_text(text, size=400):
    words = text.split()
    return [' '.join(words[i:i + size]) for i in range(0, len(words), size)]


# ============================
# 📄 LOAD DATA
# ============================
chunks = []
print("[INFO] Reading PDFs...")

for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        path = os.path.join(DATA_DIR, file)
        print(f"  → Extracting {file}")
        text = extract_pdf_text(path)
        cleaned = clean_text(text)
        for c in chunk_text(cleaned):
            chunks.append({"source": file, "text": c})

df = pd.DataFrame(chunks)
print(f"[INFO] Loaded {len(df)} chunks from {len(os.listdir(DATA_DIR))} PDFs.")
print(df.head())


# ============================
# 🧮 TF-IDF VECTORIZATION
# ============================
print("[INFO] Building vectorizer...")
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
vectorizer.fit(df["text"])  # fit once globally for consistent vocab


# ============================
# 📊 CASE-BY-CASE CLUSTERING
# ============================
print("\n[INFO] Generating case-by-case PCA plots...")

for case_name, group_df in df.groupby("source"):
    print(f"  → Processing {case_name}...")

    # Vectorize only that case's text
    X_case = vectorizer.transform(group_df["text"])

    # KMeans per case (adjust n_clusters if needed)
    n_clusters = 5
    kmeans_case = KMeans(n_clusters=n_clusters, random_state=42)
    group_df["cluster_case"] = kmeans_case.fit_predict(X_case)

    # Find top words for each cluster
    terms = vectorizer.get_feature_names_out()
    cluster_labels = []
    for i in range(n_clusters):
        centroid = kmeans_case.cluster_centers_[i]
        top_terms = [terms[j] for j in np.argsort(centroid)[-5:]]  # top 5 words
        label = ", ".join(top_terms)
        cluster_labels.append(label)

    # Reduce to 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_case.toarray())

    # Plot clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1],
                          c=group_df["cluster_case"],
                          cmap="tab10", s=50, alpha=0.7,
                          edgecolors="k", linewidth=0.3)

    # Label clusters with top words
    centroids_2d = pca.transform(kmeans_case.cluster_centers_)
    for i, (x, y) in enumerate(centroids_2d):
        plt.text(x, y, cluster_labels[i],
                 fontsize=9, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    # Title formatting
    title = os.path.splitext(case_name)[0].replace("_", " ").title()
    plt.title(f"{title}", fontsize=14, weight="bold", pad=15)
    plt.xlabel("PCA Dimension 1", fontsize=11)
    plt.ylabel("PCA Dimension 2", fontsize=11)
    plt.grid(alpha=0.2)

    # Save the figure
    save_path = os.path.join(RESULTS_DIR, f"{title}_clusters.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"     Saved: {save_path}")

    plt.show()


# ============================
# 💾 SAVE CLUSTER DATA
# ============================
output_path = os.path.join(RESULTS_DIR, "casewise_clusters.csv")
df.to_csv(output_path, index=False)
print(f"\n✅ Cluster assignments saved to {output_path}")
print("[DONE]")
