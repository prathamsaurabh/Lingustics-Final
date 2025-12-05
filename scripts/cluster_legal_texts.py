"""
cluster_legal_texts.py
---------------------------------
Clusters full legal PDFs (Brown v. Board, Roe v. Wade, etc.)
into semantic groups using TF-IDF + K-Means.
Outputs a CSV of results and prints top context words per cluster.
"""

import os
import re
import pdfplumber
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
# 🧹 TEXT PROCESSING FUNCTIONS
# ============================
def extract_pdf_text(path):
    """Extract all text from a PDF file."""
    text = ""
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text


def clean_text(text):
    """Lowercase, remove punctuation & extra spaces."""
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_text(text, size=400):
    """Split long text into ~size-word chunks."""
    words = text.split()
    return [' '.join(words[i:i + size]) for i in range(0, len(words), size)]


# ============================
# 📄 LOAD & PREPARE DATASET
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
print(f"[INFO] Loaded {len(df)} text chunks from {len(os.listdir(DATA_DIR))} PDFs.")
print(df.head())


# ============================
# 🧮 VECTORIZATION & CLUSTERING
# ============================
print("[INFO] Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X = vectorizer.fit_transform(df['text'])

k = 5  # you can change this based on number of cases
print(f"[INFO] Running K-Means with k={k}...")
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(X)


# ============================
# 🔍 TOP WORDS PER CLUSTER
# ============================
terms = vectorizer.get_feature_names_out()
print("\n========== TOP WORDS BY CLUSTER ==========")
for i in range(k):
    centroid = kmeans.cluster_centers_[i]
    top_terms = [terms[j] for j in np.argsort(centroid)[-12:]]
    print(f"\nCluster {i}: {', '.join(top_terms)}")


# ============================
# 📊 IMPROVED VISUALIZATION WITH COLOR COORDINATION & LEGEND
# ============================

from matplotlib.lines import Line2D

print("\n[INFO] Generating labeled PCA scatter plot...")

# Reduce to 2D for plotting
pca = PCA(n_components=2)
coords = pca.fit_transform(X.toarray())

# Assign a distinct color to each cluster
colors = plt.cm.get_cmap("tab10", k)  # tab10 = 10 distinct colors
cluster_colors = [colors(label) for label in df['cluster']]

plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], c=cluster_colors, alpha=0.7, s=50, edgecolors='k', linewidth=0.3)

# Label cluster centers
centroids_2d = pca.transform(kmeans.cluster_centers_)
for i, (x, y) in enumerate(centroids_2d):
    plt.text(x, y, f"Cluster {i}", fontsize=10, weight='bold',
             ha='center', va='center', color='black',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
           markerfacecolor=colors(i), markersize=8)
    for i in range(k)
]
plt.legend(handles=legend_elements, title="Cluster Key", loc='best', fontsize=9)

# Titles & labels
plt.title("Clusters of Legal Case Texts with Context Key", fontsize=14, pad=15)
plt.xlabel("PCA Dimension 1", fontsize=11)
plt.ylabel("PCA Dimension 2", fontsize=11)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()



# ============================
# 💾 SAVE RESULTS
# ============================
output_path = os.path.join(RESULTS_DIR, "legal_clusters.csv")
df.to_csv(output_path, index=False)
print(f"\n✅ Results saved to {output_path}")
print("[DONE]")
