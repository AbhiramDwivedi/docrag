import os
import openai
from github import Github
import re
import base64

REPO = os.environ.get("GITHUB_REPOSITORY")
PR_NUMBER = os.environ.get("PR_NUMBER")
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Load Copilot instructions
with open(".github/copilot-instructions.md", "r", encoding="utf-8") as f:
    instructions = f.read()

g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO)
pr = repo.get_pull(int(PR_NUMBER))
diff = pr.diff()

# Try to find a linked issue (by convention: fixes #123 or closes #123 in PR body)
issue_number = None
if pr.body:
    match = re.search(r'(?:fixes|closes)\s+#(\d+)', pr.body, re.IGNORECASE)
    if match:
        issue_number = int(match.group(1))

issue_text = ""
if issue_number:
    try:
        issue = repo.get_issue(number=issue_number)
        issue_text = f"--- Linked Issue #{issue_number} ---\nTitle: {issue.title}\nBody:\n{issue.body}\n"
    except Exception as e:
        issue_text = f"--- Linked Issue #{issue_number} could not be loaded: {e} ---\n"

# Helpers
MAX_FILE_CHARS = 4000
HEAD_REF = pr.head.sha

def fetch_text_file(path: str, ref: str) -> str:
    try:
        contents = repo.get_contents(path, ref=ref)
        if isinstance(contents, list):
            return ""
        if contents.encoding == "base64":
            data = base64.b64decode(contents.content).decode("utf-8", errors="replace")
        else:
            data = contents.decoded_content.decode("utf-8", errors="replace")
        if len(data) > MAX_FILE_CHARS:
            half = MAX_FILE_CHARS // 2
            return data[:half] + "\n...\n[truncated]...\n" + data[-half:]
        return data
    except Exception:
        return "[unavailable or binary]"

# Summarize changed files (paths and truncated head contents)
changed_files_api = list(pr.get_files())
changed_files = [f.filename for f in changed_files_api]
files_list = "\n".join(f"- {name}" for name in changed_files)
changed_files_blobs = []
for f in changed_files:
    content = fetch_text_file(f, HEAD_REF)
    changed_files_blobs.append(f"### {f}\n\n```\n{content}\n```\n")
changed_files_section = "\n".join(changed_files_blobs)

# Curated anchor files (do not fetch unless requested)
ANCHORS = [
    "backend/shared/config.py",
    "backend/shared/logging_config.py",
    "backend/ingestion/processors/embedder.py",
    "backend/ingestion/storage/vector_store.py",
    "backend/querying/agents/plugins/plugin.py",
    "backend/querying/agents/plugins/semantic_search.py",
    "backend/querying/agents/plugins/metadata_commands.py",
    "backend/main.py",
    "cli/ask.py",
    "interface/cli/ask.py",
    "tests/",
    ".github/workflows/",
    "backend/shared/config.yaml.template",
]

# First pass: ask the LLM which anchors (max 3) it wants to inspect
selection_prompt = f"""
You will review a PR for a Python RAG repository. You will first request up to 3 additional anchor files from the curated list below that you need to inspect to perform a thorough review. Choose only those that materially impact your ability to assess correctness/completeness/design of the changes.

Curated anchors (choose up to 3 paths exactly as written):
{os.linesep.join('- ' + a for a in ANCHORS)}

PR Metadata:
- Title: {pr.title}
- Body: {pr.body or ''}
- Changed files:
{files_list}

Guidance:
- If config-related changes are present, prefer backend/shared/config.py and backend/shared/config.yaml.template.
- For retrieval logic, consider semantic_search.py and vector_store.py.
- For embeddings, consider embedder.py.
- For CLI/entrypoints, consider cli/ask.py or interface/cli/ask.py.
- For CI/test impact, consider tests/ and .github/workflows/.

Respond with a JSON object of the form: {{"anchors": ["path1", "path2"]}}. If none needed, return {{"anchors": []}}.
"""

openai.api_key = OPENAI_API_KEY
sel = openai.ChatCompletion.create(
    model="gpt-5-nano",
    messages=[
        {"role": "system", "content": "You are a pragmatic code reviewer that selects minimal extra context to inspect."},
        {"role": "user", "content": selection_prompt},
    ],
    max_tokens=200,
    temperature=0.0,
)
sel_text = sel["choices"][0]["message"]["content"]
requested = []
try:
    import json
    requested = json.loads(sel_text).get("anchors", [])[:3]
except Exception:
    requested = []

# Fetch requested anchors (text only, truncated)
anchor_sections = []
for path in requested:
    blob = fetch_text_file(path, HEAD_REF)
    anchor_sections.append(f"### {path}\n\n```\n{blob}\n```\n")
anchors_context = "\n".join(anchor_sections)

SYSTEM_PROMPT = (
    "You are an expert code reviewer for this repository. "
    "Your review must ensure: correctness, completeness (meets linked issue requirements), "
    "sound design (maintainable, aligned with project architecture), security/privacy compliance, and deterministic tests/CI. "
    "Enforce the repository rules provided. "
    "Output only specific, actionable findings with concrete suggestions. "
    "Do not include praise, generic statements, or scores. "
    "If nothing actionable is found, respond with exactly: 'No actionable issues found.' "
    "When appropriate, recommend the smallest viable fix with corresponding tests; "
    "when design or architectural issues prevent correctness/completeness or violate repo rules, propose broader changes and outline a feasible plan (key changes, migration/tests)."
)

prompt = f"""
Review this pull request against the repository rules and the linked issue (if any).

Checks to perform (apply only if relevant to the changes):
- Tests: new/updated tests present; deterministic (seeded, CPU, normalized vectors, stable sorting); no network access
- CI: workflow updated if needed; all tests expected to pass in CI (not just locally)
- Config: no hardcoded paths; settings read via shared.config.Config; update config.yaml.template and docs when adding config
- Retrieval guidelines: cosine with normalized vectors; MMR with deterministic tie-break; hybrid lexical search only via separate plugin; structured debug logging behind config
- Storage/data paths: configurable and default outside the repo
- Security/privacy: no secrets committed; no data sent externally without a feature flag and documentation
- Design: simple, maintainable; respects existing architecture (ingestion → storage → querying/plugins → CLI/API)
- Completeness: fully addresses all requirements/acceptance criteria from the linked issue

--- PR Metadata ---
Title: {pr.title}
Body: {pr.body or ''}

--- Changed Files (truncated contents) ---
{changed_files_section}

{issue_text}
--- Repository Rules (Copilot Instructions) ---
{instructions}

--- Additional Context (requested anchors) ---
{anchors_context or '[none requested]'}

--- PR Diff ---
{diff}

Provide only actionable findings as bullet points. For each, include: file path (and line/symbol if clear) and a concrete fix or test to add. If proposing broader changes, include a short plan with the minimum necessary steps. If none, reply: No actionable issues found.
"""

review_resp = openai.ChatCompletion.create(
    model="gpt-5-mini",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ],
    max_tokens=1200,
    temperature=0.2,
)
review = review_resp["choices"][0]["message"]["content"]

pr.create_issue_comment(f"## Copilot LLM Review\n{review}")
