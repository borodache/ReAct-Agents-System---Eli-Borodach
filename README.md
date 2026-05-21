# Bitext Dataset Agent (Nebius Assignment 3)

A LangGraph **ReAct agent** that answers questions about the [Bitext customer-support training dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), plus an optional **MCP server** that exposes core dataset tools over HTTP or stdio.

The agent routes each question to structured tools (counts, categories, examples), unstructured summarization tools, profile recall, or an out-of-scope decline path. Conversation history is checkpointed in SQLite; user facts are stored in `.profiles/`.

## Features

- **Query router** — classifies questions as structured, unstructured, profile recall, or out-of-scope
- **ReAct loop** — tool calling with natural-language answers (via `answer_format.py`)
- **Session memory** — SQLite checkpoints under `.checkpoints/` (`--session`)
- **User profiles** — persistent facts in `.profiles/` (`--user`), updated from chat
- **MCP server** — `mcp_server.py` exposes four dataset tools with plain-English responses

## Requirements

- Python 3.11+ (tested on 3.13)
- [Nebius Token Factory](https://studio.nebius.ai/) API key
- Dependencies in `requirements.txt`

## Setup

```powershell
cd "path\to\Nebius-Agents-Assignment-3"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Credentials

**Without a `.env` file** — set system environment variables:

```powershell
$env:NEBIUS_API_KEY = "your_api_key_here"
$env:NEBIUS_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
```

**With a `.env` file** (project root) — values in `.env` **override** the environment variables above:

```env
NEBIUS_API_KEY=your_api_key_here
NEBIUS_MODEL=meta-llama/Llama-3.3-70B-Instruct
# Optional: separate model for profile Q&A / updates
# NEBIUS_PROFILE_MODEL=meta-llama/Llama-3.3-70B-Instruct
# Optional: faster Hugging Face dataset downloads
# HF_TOKEN=your_hf_token
```

Use a model that supports **tool calling** on Nebius (e.g. Llama 3.3 70B). Smaller models such as Llama 3.1 8B are remapped automatically but are not recommended.

## CLI agent (`main.py`)

Interactive chat (default session and user id: `default`):

```powershell
python main.py
```

With session and user profile ids:

```powershell
python main.py --session 0 --user 0
```

Single question:

```powershell
python main.py --session 0 "How many refund requests did we get?"
```

Quiet mode (final answer only):

```powershell
python main.py --quiet --session 0 "What categories exist?"
```

Run bundled example questions:

```powershell
python main.py --examples
python main.py --structured-examples
python main.py --unstructured-examples
python main.py --out-of-scope-examples
```

| Flag | Description |
|------|-------------|
| `--session ID` | Conversation checkpoint id (restores history after restart) |
| `--user ID` | Profile file in `.profiles/` (defaults to session id) |
| `--model MODEL` | Override `NEBIUS_MODEL` |
| `--quiet` | Hide router / tool trace |
| `--examples` | Run all eight demo questions and exit |

Type `quit`, `exit`, or `q` to leave interactive mode.

## Streamlit UI (`streamlit_app.py`)

Web chat interface with the same agent, sessions, and user profiles:

```powershell
streamlit run streamlit_app.py
```

Use the sidebar to set **Session ID** and **User ID**, toggle **Show reasoning trace**, or click example questions. Chat history in the UI resets when you change session/user; the agent still restores prior turns from SQLite checkpoints for the same session id.

## MCP server (`mcp_server.py`)

HTTP is the **default** transport (for `fastmcp call` and remote clients).

**Terminal 1 — start server:**

```powershell
python mcp_server.py
```

Server URL: `http://127.0.0.1:8000/mcp`

Optional flags: `--host`, `--port` (e.g. `--port 8001` if 8000 is busy).

**Terminal 2 — call tools:**

```powershell
fastmcp list http://localhost:8000/mcp
fastmcp call http://localhost:8000/mcp get_dataset_categories
fastmcp call http://localhost:8000/mcp count_dataset_records category=ORDER
fastmcp call http://localhost:8000/mcp get_dataset_examples limit=3 category=SHIPPING
fastmcp call http://localhost:8000/mcp filter_by_intent intent_contains=refund
```

For **Cursor / Claude Desktop** (stdio transport):

```powershell
python mcp_server.py --stdio
```

Example Cursor MCP config (stdio):

```json
{
  "mcpServers": {
    "bitext": {
      "command": "python",
      "args": ["C:/path/to/Nebius-Agents-Assignment-3/mcp_server.py", "--stdio"]
    }
  }
}
```

Example HTTP config:

```json
{
  "mcpServers": {
    "bitext": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### MCP tools

| Tool | Purpose |
|------|---------|
| `get_dataset_categories` | List categories with row counts |
| `count_dataset_records` | Count rows (optional category / intent filters) |
| `filter_by_intent` | Save an intent filter; returns `filter_id` for chaining |
| `get_dataset_examples` | Sample example conversations |

Tool responses are returned as **natural language** (not raw JSON).

### Port already in use

If you see `WinError 10048` on port 8000, another server instance is still running:

```powershell
netstat -ano | findstr ":8000"
taskkill /PID <pid> /F
```

Or start on another port: `python mcp_server.py --port 8001`

## Project layout

```
├── main.py           # CLI entry point
├── agent.py          # LangGraph ReAct graph + router
├── router.py         # Query classification
├── tools.py          # Dataset tool implementations
├── tool_schemas.py   # Pydantic inputs/outputs
├── answer_format.py  # JSON → natural language
├── dataset_store.py  # Hugging Face dataset cache
├── config.py         # .env and Nebius model setup
├── user_profile.py   # Profile load/save/update
├── checkpointer.py   # SQLite conversation checkpoints
├── filter_context.py # In-memory filter_id chain (per process)
├── mcp_server.py     # FastMCP HTTP/stdio server
├── streamlit_app.py  # Streamlit web UI
└── requirements.txt
```

Generated/local data (gitignored): `.env`, `.checkpoints/`, `.profiles/`

## Example questions

**Structured**

- What categories exist in the dataset?
- How many refund requests did we get?
- Show me 3 examples from the SHIPPING intent.
- What is the distribution of intents in the ACCOUNT category?

**Unstructured**

- Summarize the FEEDBACK category.
- How do customer service representatives typically respond to cancellation requests?

**Out of scope**

- Who won the 2024 Champions League?
- Write me a poem about customer service.

## License / assignment

Course assignment materials may apply; see `From_AI_Model_to_AI_Agent_-_Assignment_3.pdf`.
