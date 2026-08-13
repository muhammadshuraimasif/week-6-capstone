import os
import re
import json
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# ---- CONFIG ----
OLLAMA_MODEL = "qwen2.5-coder:1.5b"
TOP_K = 5
REPO_PATH = "C:/Users/asif_/AI-Mock-interview/ai-interview-mocker"

EXCLUDE_DIRS = {
    ".git", "node_modules", "venv", "__pycache__", ".venv",
    ".next", "dist", "build", ".cache", "out", "coverage",
    ".vercel", ".turbo", "public"
}
CODE_EXTENSIONS = (".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".html", ".css")

# ---- SETUP ----
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="codebase")

print(f"Ready. Loaded collection with {collection.count()} chunks.")
print("Ask questions about your codebase. Type 'exit' to quit.\n")


# =========================================================
# TOOLS
# =========================================================

def tool_search_codebase(query, top_k=TOP_K):
    """Semantic search over code chunks. Good for 'how does X work' style questions."""
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    blocks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        blocks.append(f"File: {meta['file']} (lines {meta['start_line']}-{meta['end_line']})\n```\n{doc}\n```")
    return "\n\n".join(blocks), [m["file"] for m in results["metadatas"][0]]


def tool_grep_exact(term):
    """Exact text search across the repo. Good for finding TODOs, specific strings, exact function names."""
    matches = []
    for root, dirs, files in os.walk(REPO_PATH):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for file in files:
            if not file.endswith(CODE_EXTENSIONS):
                continue
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if term.lower() in line.lower():
                            rel = os.path.relpath(filepath, REPO_PATH)
                            matches.append(f"{rel}:{i}: {line.strip()}")
            except Exception:
                continue
            if len(matches) >= 30:  # cap results
                break

    if not matches:
        return f"No exact matches found for '{term}'.", []
    return "\n".join(matches[:30]), []


def tool_list_files():
    """List all files in the project, grouped by folder."""
    lines = []
    for root, dirs, files in os.walk(REPO_PATH):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        code_files = [f for f in files if f.endswith(CODE_EXTENSIONS)]
        if code_files:
            rel_root = os.path.relpath(root, REPO_PATH)
            lines.append(f"{rel_root}/: " + ", ".join(code_files))
    return "\n".join(lines), []


TOOLS = {
    "search_codebase": tool_search_codebase,
    "grep_exact": tool_grep_exact,
    "list_files": tool_list_files,
}

TOOL_DESCRIPTIONS = """Available tools:
1. search_codebase(query) - Semantic search. Use for conceptual questions like "how does X work", "explain the login flow".
2. grep_exact(term) - Exact keyword search. Use for finding specific strings like "TODO", "console.log", exact function/variable names.
3. list_files() - Lists every file in the project grouped by folder. Use for questions about overall project structure or "what files exist"."""


# =========================================================
# ROUTING: ask the model which tool to use
# =========================================================

def choose_tool(question):
    routing_prompt = f"""{TOOL_DESCRIPTIONS}

Given the user's question, pick exactly ONE tool and respond with ONLY a JSON object, nothing else.
Format: {{"tool": "tool_name", "input": "argument or empty string"}}

Question: {question}

JSON:"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": routing_prompt}]
    )
    raw = response["message"]["content"].strip()

    # try to extract JSON even if model adds extra text
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if parsed.get("tool") in TOOLS:
                return parsed["tool"], parsed.get("input", "")
        except json.JSONDecodeError:
            pass

    # fallback if model output was unusable
    print(f"  (routing fallback triggered — raw model output was: {raw[:100]})")
    return "search_codebase", question


# =========================================================
# MAIN ASK FLOW
# =========================================================

def ask(question):
    tool_name, tool_input = choose_tool(question)
    print(f"\n[agent chose tool: {tool_name}({tool_input!r})]")

    tool_fn = TOOLS[tool_name]
    if tool_name == "list_files":
        context, sources = tool_fn()
    else:
        context, sources = tool_fn(tool_input if tool_input else question)

    if sources:
        print("--- Sources used ---")
        for s in sources:
            print(f"  {s}")
        print("---------------------")

    final_prompt = f"""You are a helpful assistant answering questions about a codebase.
Use ONLY the information below to answer. If the answer isn't there, say so clearly instead of guessing.

INFORMATION:
{context}

QUESTION:
{question}

ANSWER:"""

    print("\nThinking...\n")
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": final_prompt}]
    )
    print(response["message"]["content"])
    print()


if __name__ == "__main__":
    while True:
        question = input("Ask> ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        ask(question)