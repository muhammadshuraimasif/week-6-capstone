import os
import re
import json
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.status import Status
from rich.prompt import Prompt
from rich import box

console = Console()

# ---- CONFIG ----
OLLAMA_MODEL = "qwen2.5-coder:1.5b"
TOP_K = 5
MAX_HISTORY_TURNS = 6  # how many past Q&A pairs to keep in memory

REPO_PATHS = {
    "interview-mocker": "C:/Users/asif_/AI-Mock-interview/ai-interview-mocker",
    # "prescripto": "C:/Users/asif_/Projects/prescripto",
}

EXCLUDE_DIRS = {
    ".git", "node_modules", "venv", "__pycache__", ".venv",
    ".next", "dist", "build", ".cache", "out", "coverage",
    ".vercel", ".turbo", "public"
}
CODE_EXTENSIONS = (".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".html", ".css")

# conversation memory: list of {"question": ..., "answer": ...}
conversation_history = []


# =========================================================
# SETUP
# =========================================================

def setup():
    with console.status("[bold cyan]Loading embedding model...", spinner="dots"):
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="codebase")

    existing = collection.get(limit=collection.count())
    repos_in_db = sorted(set(m.get("repo", "unknown") for m in existing["metadatas"]))

    return embedder, collection, repos_in_db


# =========================================================
# TOOLS
# =========================================================

def tool_search_codebase(embedder, collection, query, repo_filter, top_k=TOP_K):
    query_embedding = embedder.encode([query]).tolist()
    where = {"repo": repo_filter} if repo_filter else None
    results = collection.query(query_embeddings=query_embedding, n_results=top_k, where=where)

    blocks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        blocks.append(
            f"[{meta.get('repo', '?')}] File: {meta['file']} (lines {meta['start_line']}-{meta['end_line']})\n```\n{doc}\n```"
        )
    sources = [f"[{m.get('repo', '?')}] {m['file']}" for m in results["metadatas"][0]]
    return "\n\n".join(blocks), sources


def tool_grep_exact(term, repo_filter):
    matches = []
    repos_to_search = [repo_filter] if repo_filter else list(REPO_PATHS.keys())

    for repo_name in repos_to_search:
        repo_path = REPO_PATHS.get(repo_name)
        if not repo_path or not os.path.isdir(repo_path):
            continue
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
            for file in files:
                if not file.endswith(CODE_EXTENSIONS):
                    continue
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if term.lower() in line.lower():
                                rel = os.path.relpath(filepath, repo_path)
                                matches.append(f"[{repo_name}] {rel}:{i}: {line.strip()}")
                except Exception:
                    continue
                if len(matches) >= 30:
                    break

    if not matches:
        return f"No exact matches found for '{term}'.", []
    return "\n".join(matches[:30]), []


def tool_list_files(repo_filter):
    repos_to_list = [repo_filter] if repo_filter else list(REPO_PATHS.keys())
    lines = []
    for repo_name in repos_to_list:
        repo_path = REPO_PATHS.get(repo_name)
        if not repo_path or not os.path.isdir(repo_path):
            continue
        lines.append(f"=== {repo_name} ===")
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
            code_files = [f for f in files if f.endswith(CODE_EXTENSIONS)]
            if code_files:
                rel_root = os.path.relpath(root, repo_path)
                lines.append(f"{rel_root}/: " + ", ".join(code_files))
    return "\n".join(lines), []


TOOL_DESCRIPTIONS = """Available tools:
1. search_codebase(query) - Semantic search. Use for conceptual questions like "how does X work", "explain the login flow".
2. grep_exact(term) - Exact keyword search. Use for finding specific strings like "TODO", exact function/variable names.
3. list_files() - Lists every file in the project grouped by folder. Use for structure questions."""


def choose_tool(question, history_text=""):
    history_note = f"\n{history_text}\n(Use this history to resolve pronouns like 'that', 'it', 'this' in the question into a concrete search term.)\n" if history_text else ""

    routing_prompt = f"""{TOOL_DESCRIPTIONS}
{history_note}
Given the user's question, pick exactly ONE tool and respond with ONLY a JSON object, nothing else.
The "input" should be a self-contained, concrete search term — resolve any pronouns using the conversation history above.
Format: {{"tool": "tool_name", "input": "argument or empty string"}}

Question: {question}

JSON:"""

    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": routing_prompt}])
    raw = response["message"]["content"].strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if parsed.get("tool") in ("search_codebase", "grep_exact", "list_files"):
                return parsed["tool"], parsed.get("input", "")
        except json.JSONDecodeError:
            pass

    return "search_codebase", question


# =========================================================
# MEMORY FORMATTING
# =========================================================

def format_history():
    if not conversation_history:
        return ""
    recent = conversation_history[-MAX_HISTORY_TURNS:]
    lines = ["PREVIOUS CONVERSATION (for context, most recent last):"]
    for turn in recent:
        lines.append(f"User asked: {turn['question']}")
        lines.append(f"You answered: {turn['answer']}")
    return "\n".join(lines) + "\n"


# =========================================================
# MAIN ASK FLOW
# =========================================================

def ask(embedder, collection, question, repo_filter):
    history_block = format_history()
    tool_name, tool_input = choose_tool(question, history_block)
    label = repo_filter if repo_filter else "ALL REPOS"

    console.print(f"[dim]→ tool: [bold]{tool_name}[/bold]({tool_input!r})  |  scope: {label}[/dim]")

    if tool_name == "search_codebase":
        context, sources = tool_search_codebase(embedder, collection, tool_input if tool_input else question, repo_filter)
    elif tool_name == "grep_exact":
        context, sources = tool_grep_exact(tool_input if tool_input else question, repo_filter)
    else:
        context, sources = tool_list_files(repo_filter)

    if sources:
        console.print(f"[dim]  sources: {', '.join(sources[:5])}[/dim]")

    final_prompt = f"""You are a helpful assistant answering questions about a codebase (possibly across multiple projects, marked with [repo-name]).
Use the conversation history to understand follow-up questions (like "what about X" or "explain more").
Use ONLY the CODE INFORMATION below to answer factual questions about the code. If the answer isn't there, say so clearly instead of guessing.

{history_block}
CODE INFORMATION:
{context}

CURRENT QUESTION:
{question}

ANSWER:"""

    with console.status("[bold cyan]Thinking...", spinner="dots"):
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": final_prompt}])

    answer = response["message"]["content"]

    console.print(Panel(Markdown(answer), title="[bold green]Agent[/bold green]", border_style="green", box=box.ROUNDED))

    conversation_history.append({"question": question, "answer": answer})


# =========================================================
# REPO SELECTION
# =========================================================

def choose_repo(repos_in_db):
    console.print("\n[bold]Which repo do you want to query?[/bold]")
    for i, r in enumerate(repos_in_db, 1):
        console.print(f"  [cyan]{i}[/cyan]. {r}")
    console.print(f"  [cyan]{len(repos_in_db) + 1}[/cyan]. All repos")

    choice = Prompt.ask("Choose a number", default=str(len(repos_in_db) + 1))
    try:
        idx = int(choice) - 1
        if idx == len(repos_in_db):
            return None
        if 0 <= idx < len(repos_in_db):
            return repos_in_db[idx]
    except ValueError:
        pass
    return None


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]Codebase Q&A Agent[/bold cyan]\n[dim]Local · Free · RAG + Tool-Calling + Memory[/dim]",
        border_style="cyan", box=box.DOUBLE
    ))

    embedder, collection, repos_in_db = setup()
    console.print(f"[green]✓[/green] Loaded [bold]{collection.count()}[/bold] chunks across repos: [bold]{', '.join(repos_in_db)}[/bold]\n")

    current_repo = choose_repo(repos_in_db)
    console.print(f"\n[bold]Scope:[/bold] [yellow]{current_repo if current_repo else 'ALL REPOS'}[/yellow]")
    console.print("[dim]Type 'switch' to change repo, 'clear' to forget conversation, 'exit' to quit.[/dim]\n")

    while True:
        question = Prompt.ask("\n[bold blue]You[/bold blue]")
        if question.lower() in ("exit", "quit"):
            console.print("[dim]Goodbye![/dim]")
            break
        if question.lower() == "switch":
            current_repo = choose_repo(repos_in_db)
            console.print(f"\n[bold]Scope:[/bold] [yellow]{current_repo if current_repo else 'ALL REPOS'}[/yellow]\n")
            continue
        if question.lower() == "clear":
            conversation_history.clear()
            console.print("[dim]Conversation memory cleared.[/dim]\n")
            continue
        if not question.strip():
            continue

        ask(embedder, collection, question, current_repo)
