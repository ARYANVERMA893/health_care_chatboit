import os
import re
import textwrap
import traceback
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ============================================================
# OPENAI CONFIG
# ============================================================
OPENAI_API_KEY = os.getenv("")  # or paste key directly
GPT_MODEL = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = r"C:\Users\VICTUS\Desktop\chatbot\dataset\archive"
DATASET_CSV = os.path.join(DATA_DIR, "dataset.csv")
DESC_CSV = os.path.join(DATA_DIR, "symptom_Description.csv")
PREC_CSV = os.path.join(DATA_DIR, "symptom_precaution.csv")
SEV_CSV = os.path.join(DATA_DIR, "Symptom-severity.csv")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_GEN_TOKENS = 256
TOP_K = 1

PROMPT_TEMPLATE = """You are a cautious medical assistant.

User symptoms:
{user_symptoms}

Retrieved medical knowledge:
{retrieved_texts}

Tasks:
- List top possible diseases (short list).
- Give 1–2 lines of reasoning for each.
- Give a 1-line description.
- Give up to 3 precautions.
- End with a disclaimer that this is not a diagnosis.

Answer clearly and concisely.
"""

# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    ds = pd.read_csv(DATASET_CSV)
    desc = pd.read_csv(DESC_CSV)
    prec = pd.read_csv(PREC_CSV)
    sev = pd.read_csv(SEV_CSV)
    return ds, desc, prec, sev

# ============================================================
# DOCUMENT BUILDING
# ============================================================
def build_docs_dataframe(dataset_df, desc_df, prec_df):
    docs = []
    for _, row in dataset_df.iterrows():
        disease = str(row.iloc[0]).strip()
        syms = [
            str(v).strip().lower()
            for v in row.iloc[1:].values
            if pd.notna(v) and str(v).strip() != ""
        ]

        ddesc = desc_df.loc[
            desc_df.iloc[:, 0].str.lower() == disease.lower()
        ]
        description = ddesc.iloc[0, 1] if len(ddesc) > 0 else ""

        p = prec_df.loc[
            prec_df.iloc[:, 0].str.lower() == disease.lower()
        ]
        precautions = []
        if len(p) > 0:
            rowp = p.iloc[0, 1:]
            precautions = [
                str(x).strip()
                for x in rowp.values
                if pd.notna(x) and str(x).strip() != ""
            ]

        docs.append({
            "disease": disease,
            "symptoms": syms,
            "description": description,
            "precautions": precautions
        })

    return pd.DataFrame(docs)

def doc_to_text(doc):
    s = f"Disease: {doc['disease']}. Symptoms: {', '.join(doc['symptoms'])}."
    if doc.get("description"):
        s += f" Description: {doc['description']}"
    if doc.get("precautions"):
        s += f" Precautions: {', '.join(doc['precautions'])}."
    return s
def initialize_chatbot():
    print("Loading data and building index...")
    ds, desc, prec, sev = load_data()
    docs_df = build_docs_dataframe(ds, desc, prec).reset_index(drop=True)
    index, _ = create_faiss_index(docs_df)
    print("Chatbot ready.")
    return docs_df, index

# ============================================================
# FAISS INDEX
# ============================================================
def create_faiss_index(docs_df, index_path="faiss.index", force_rebuild=False):
    if os.path.exists(index_path) and not force_rebuild:
        print("Loading existing FAISS index...")
        index = faiss.read_index(index_path)
        embeddings = np.load(index_path + ".emb.npy")
        return index, embeddings

    print("Building FAISS index...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [doc_to_text(r) for _, r in docs_df.iterrows()]
    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, index_path)
    np.save(index_path + ".emb.npy", emb)

    return index, emb

def retrieve_topk(user_symptoms, docs_df, index, k=TOP_K):
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    query = "Symptoms: " + ", ".join(user_symptoms)

    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, k)
    results = []

    for idx, score in zip(I[0], D[0]):
        if idx >= 0:
            results.append((docs_df.iloc[idx].to_dict(), float(score)))

    return results

def build_context_text(retrieved):
    texts = []
    for doc, score in retrieved:
        texts.append(
            f"Disease: {doc['disease']}\n"
            f"Symptoms: {', '.join(doc['symptoms'])}\n"
            f"Description: {doc['description']}\n"
            f"Precautions: {', '.join(doc['precautions'])}\n"
        )
    return "\n\n".join(texts)

# ============================================================
# GPT RESPONSE
# ============================================================
def generate_response(user_symptoms, retrieved):
    prompt = PROMPT_TEMPLATE.format(
        user_symptoms=", ".join(user_symptoms),
        retrieved_texts=build_context_text(retrieved)
    )

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a cautious medical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=MAX_GEN_TOKENS
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        traceback.print_exc()
        return f"⚠️ GPT API error: {str(e)}"

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    print("Loading datasets...")
    ds, desc, prec, sev = load_data()

    docs_df = build_docs_dataframe(ds, desc, prec).reset_index(drop=True)
    print(f"{len(docs_df)} disease documents loaded.")

    index, _ = create_faiss_index(docs_df)

    print("\n--- Healthcare RAG Chatbot ---")

    while True:
        user_symptoms = []

        print("\nEnter symptoms one by one (type 'done'):")
        while True:
            s = input("Symptom: ").strip().lower()
            if s == "done":
                break
            s = re.sub(r"[^a-z0-9\s]", "", s)
            if s:
                user_symptoms.append(s)

        if not user_symptoms:
            print("No symptoms entered. Exiting.")
            break

        retrieved = retrieve_topk(user_symptoms, docs_df, index)

        print("\nGenerating response...\n")
        response = generate_response(user_symptoms, retrieved)

        print("-" * 60)
        print(textwrap.fill(response, width=100))
        print("-" * 60)

        if input("Analyze another? (yes/no): ").lower() not in ("yes", "y"):
            break

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
