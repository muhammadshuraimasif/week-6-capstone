import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# ---- CONFIG ----
OLLAMA_MODEL = "qwen2.5-coder:1.5b"
TOP_K = 5  # how many chunks to retrieve per question

# ---- SETUP ----
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="codebase")

print(f"Ready. Loaded collection with {collection.count()} chunks.")
print("Ask questions about your codebase. Type 'exit' to quit.\n")


def retrieve(question, top_k=TOP_K):
    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results


def build_prompt(question, results):
    context_blocks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_blocks.append(
            f"File: {meta['file']} (lines {meta['start_line']}-{meta['end_line']})\n```\n{doc}\n```"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant answering questions about a codebase.
Use ONLY the code context below to answer. If the answer isn't in the context, say so.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    return prompt


def ask(question):
    results = retrieve(question)

    if not results["documents"][0]:
        print("No relevant code found for that question.\n")
        return

    prompt = build_prompt(question, results)

    print("\n--- Sources used ---")
    for meta in results["metadatas"][0]:
        print(f"  {meta['file']} (lines {meta['start_line']}-{meta['end_line']})")
    print("---------------------\n")

    print("Thinking...\n")
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    print(response["message"]["content"])
    print()


# ---- MAIN LOOP ----
if __name__ == "__main__":
    while True:
        question = input("Ask> ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        ask(question)