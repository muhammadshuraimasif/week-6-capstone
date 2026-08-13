import os
import chromadb
from sentence_transformers import SentenceTransformer

# ---- CONFIG ----
# Add as many repos as you want here. "name" is how you'll refer to it when querying.
REPOS = [
    {"name": "interview-mocker", "path": "C:/Users/asif_/AI-Mock-interview/ai-interview-mocker"},
    # {"name": "prescripto", "path": "C:/Users/asif_/Projects/prescripto"},
    # add more repos here as needed
]

CODE_EXTENSIONS = (".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".html", ".css")
CHUNK_SIZE = 50
BATCH_SIZE = 100

EXCLUDE_DIRS = {
    ".git", "node_modules", "venv", "__pycache__", ".venv",
    ".next", "dist", "build", ".cache", "out", "coverage",
    ".vercel", ".turbo", "public"
}
EXCLUDE_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
MAX_FILE_SIZE_BYTES = 300_000

# ---- SETUP ----
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection(name="codebase")
except Exception:
    pass
collection = client.get_or_create_collection(name="codebase")


def chunk_file(filepath, content):
    lines = content.split("\n")
    chunks = []
    for i in range(0, len(lines), CHUNK_SIZE):
        chunk_lines = lines[i:i + CHUNK_SIZE]
        chunk_text = "\n".join(chunk_lines)
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "file": filepath,
                "start_line": i + 1,
                "end_line": min(i + CHUNK_SIZE, len(lines))
            })
    return chunks


def build_structure_overview(repo_path):
    lines = ["PROJECT FOLDER STRUCTURE:"]
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        depth = root.replace(repo_path, "").count(os.sep)
        indent = "  " * depth
        folder_name = os.path.basename(root) or os.path.basename(repo_path)
        lines.append(f"{indent}{folder_name}/")
        sub_indent = "  " * (depth + 1)
        for f in files:
            if f.endswith(CODE_EXTENSIONS) and f not in EXCLUDE_FILES:
                lines.append(f"{sub_indent}{f}")
    return "\n".join(lines)


def ingest_repo(repo_name, repo_path):
    print(f"\n=== Ingesting repo: {repo_name} ({repo_path}) ===")

    if not os.path.isdir(repo_path):
        print(f"  SKIPPED: path does not exist.")
        return []

    chunks = []

    structure_text = build_structure_overview(repo_path)
    chunks.append({
        "text": structure_text,
        "file": "PROJECT_STRUCTURE_OVERVIEW",
        "start_line": 1,
        "end_line": structure_text.count("\n") + 1
    })

    skipped_large = 0
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for file in files:
            if file in EXCLUDE_FILES or not file.endswith(CODE_EXTENSIONS):
                continue
            filepath = os.path.join(root, file)
            try:
                if os.path.getsize(filepath) > MAX_FILE_SIZE_BYTES:
                    skipped_large += 1
                    continue
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                relative_path = os.path.relpath(filepath, repo_path)
                chunks.extend(chunk_file(relative_path, content))
            except Exception as e:
                print(f"  Skipping {filepath}: {e}")

    print(f"  Found {len(chunks)} chunks (skipped {skipped_large} large files)")

    # tag every chunk with which repo it came from
    for c in chunks:
        c["repo"] = repo_name

    return chunks


# ---- INGEST ALL REPOS ----
all_chunks = []
for repo in REPOS:
    all_chunks.extend(ingest_repo(repo["name"], repo["path"]))

print(f"\nTotal chunks across all repos: {len(all_chunks)}")

# ---- EMBED AND STORE IN BATCHES ----
if all_chunks:
    print("Generating embeddings and storing in batches...")

    for start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[start:start + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        embeddings = embedder.encode(texts).tolist()

        ids = [f"chunk_{start + i}" for i in range(len(batch))]
        metadatas = [
            {"file": c["file"], "start_line": c["start_line"], "end_line": c["end_line"], "repo": c["repo"]}
            for c in batch
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(f"  Stored {min(start + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)} chunks...")

    print(f"\nDone! Stored {len(all_chunks)} chunks from {len(REPOS)} repo(s) in ChromaDB.")
else:
    print("No chunks found. Check your REPOS config.")
