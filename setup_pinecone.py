import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()  # reads .env at project root

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in .env")

if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME is not set in .env")

# 1) Init Pinecone client (env is handled server‑side in v3)
pc = Pinecone(api_key=PINECONE_API_KEY)  # [web:54]

# 2) Create index if it does not exist yet
# We will use 384‑dim embeddings from all-MiniLM-L6-v2. [web:4][web:6]
dimension = 384

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating index '{PINECONE_INDEX_NAME}' ...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",  # match your Pinecone region
        ),
    )
else:
    print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

# Wait until index is ready and get handle
index = pc.Index(PINECONE_INDEX_NAME)
print("Index status:", index.describe_index_stats())

# 3) Prepare project documents for RAG
documents = [
    {
        "id": "problem",
        "text": (
            "Business problem: Automatically classify facility expense transactions "
            "into Services, Equipment, and Material using only the free-text Remarks "
            "field from data.xlsx. Goal is to reduce manual tagging effort and cost "
            "while keeping errors low on high-value transactions."
        ),
        "section": "problem",
    },
    {
        "id": "eda",
        "text": (
            "EDA summary: Most transactions are low value but a small number of "
            "high-value rows drive most of the total spend. Plots include volume vs "
            "value, amount boxplots, text length heatmaps, and label confidence "
            "distributions from weak supervision."
        ),
        "section": "eda",
    },
    {
        "id": "weak-supervision",
        "text": (
            "Weak supervision: Silver labels were created using labeling functions "
            "such as keyword rules on Remarks, amount-based heuristics, and a "
            "zero-shot model. These silver labels are used as training data."
        ),
        "section": "weak_supervision",
    },
    {
        "id": "baseline",
        "text": (
            "Baseline model: TF-IDF vectorizer plus Logistic Regression. It treats "
            "Remarks as a bag-of-words, is fast and interpretable but does not "
            "capture semantic similarity between phrases."
        ),
        "section": "baseline_model",
    },
    {
        "id": "setfit",
        "text": (
            "SetFit model: A few-shot classifier using the sentence-transformers "
            "all-MiniLM-L6-v2 encoder with a SetFit head, trained on silver labels. "
            "It captures sentence meaning and is more robust to label noise."
        ),
        "section": "setfit_model",
    },
    {
        "id": "business-impact",
        "text": (
            "Business impact: Using the SetFit model instead of manual tagging "
            "reduces manual review for most low-value transactions while keeping "
            "high-value accuracy high, leading to large cost and time savings."
        ),
        "section": "business_impact",
    },
]

# 4) Embed and upsert into Pinecone
print("Loading embedding model all-MiniLM-L6-v2 ...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim sentence embeddings [web:4][web:6]

texts = [d["text"] for d in documents]
embeddings = embedder.encode(texts).tolist()

vectors = []
for doc, emb in zip(documents, embeddings):
    vectors.append(
        {
            "id": doc["id"],
            "values": emb,
            "metadata": {
                "text": doc["text"],
                "section": doc["section"],
            },
        }
    )

print(f"Upserting {len(vectors)} documents into index '{PINECONE_INDEX_NAME}' ...")
index.upsert(vectors=vectors)
print("Done. You can now query this index from your app.")
