import os
import chromadb
from sentence_transformers import SentenceTransformer

# ---- CONFIG ----
REPO_PATH = "C:/Users/asif_/AI-Mock-interview/ai-interview-mocker"
CODE_EXTENSIONS = (".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".html", ".css")
CHUNK_SIZE = 50  # lines per chunk
BATCH_SIZE = 100  # how many chunks to insert into Chroma at once

# folders to completely skip
EXCLUDE_DIRS = {
    ".git", "node_modules", "venv", "__pycache__", ".venv",
    ".next", "dist", "build", ".cache", "out", "coverage",
    ".vercel", ".turbo", "public"
}

# specific files to skip (lock files, minified stuff, etc.)
EXCLUDE_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml"
}

# skip any file bigger than this (avoids huge generated/minified files)
MAX_FILE_SIZE_BYTES = 300_000  # ~300 KB

# ---- SETUP ----
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

# start fresh each time (avoids duplicate/stale chunks on re-run)
try:
    client.delete_collection(name="codebase")
except Exception:
    pass
collection = client.get_or_create_collection(name="codebase")


# ---- HELPER: chunk a file into pieces ----
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


# ---- BUILD A FOLDER STRUCTURE OVERVIEW CHUNK ----
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


# ---- WALK REPO AND COLLECT CHUNKS ----
all_chunks = []
skipped_large = 0

structure_text = build_structure_overview(REPO_PATH)
all_chunks.append({
    "text": structure_text,
    "file": "PROJECT_STRUCTURE_OVERVIEW",
    "start_line": 1,
    "end_line": structure_text.count("\n") + 1
})

for root, dirs, files in os.walk(REPO_PATH):
    # prune excluded directories in-place so os.walk doesn't descend into them
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]

    for file in files:
        if file in EXCLUDE_FILES:
            continue
        if not file.endswith(CODE_EXTENSIONS):
            continue

        filepath = os.path.join(root, file)

        try:
            if os.path.getsize(filepath) > MAX_FILE_SIZE_BYTES:
                skipped_large += 1
                continue

            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            relative_path = os.path.relpath(filepath, REPO_PATH)
            all_chunks.extend(chunk_file(relative_path, content))
        except Exception as e:
            print(f"Skipping {filepath}: {e}")

print(f"Found {len(all_chunks)} chunks from repo. (skipped {skipped_large} large files)")

# ---- EMBED AND STORE IN BATCHES ----
if all_chunks:
    print("Generating embeddings and storing in batches...")

    for start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[start:start + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        embeddings = embedder.encode(texts).tolist()

        ids = [f"chunk_{start + i}" for i in range(len(batch))]
        metadatas = [
            {"file": c["file"], "start_line": c["start_line"], "end_line": c["end_line"]}
            for c in batch
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(f"  Stored {min(start + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)} chunks...")

    print(f"Done! Stored {len(all_chunks)} chunks in ChromaDB.")
else:
    print("No code files found. Check your REPO_PATH.")