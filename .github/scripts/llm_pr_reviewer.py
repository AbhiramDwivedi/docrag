import os
from openai import OpenAI
from github import Github
import re
import base64
import requests

REPO = os.environ.get("GITHUB_REPOSITORY")
PR_NUMBER = os.environ.get("PR_NUMBER")
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Validate required environment variables
if not REPO:
    raise ValueError("GITHUB_REPOSITORY environment variable is not set")
if not PR_NUMBER:
    raise ValueError("PR_NUMBER environment variable is not set")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Load Copilot instructions
with open(".github/copilot-instructions.md", "r", encoding="utf-8") as f:
    instructions = f.read()

g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO)
pr = repo.get_pull(int(PR_NUMBER))

# Get the diff using the raw GitHub API
diff_url = f"https://api.github.com/repos/{REPO}/pulls/{PR_NUMBER}"
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3.diff"
}
diff_response = requests.get(diff_url, headers=headers)
diff = diff_response.text if diff_response.status_code == 200 else ""

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
    "README.md",
    "CONTRIBUTING.md",
    "docs/",
    "backend/pyproject.toml",
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
- For dependency changes, consider backend/pyproject.toml.
- For documentation changes or new features, consider README.md, CONTRIBUTING.md, and docs/.

Respond with a JSON object of the form: {{"anchors": ["path1", "path2"]}}. If none needed, return {{"anchors": []}}.
"""

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

sel = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a pragmatic code reviewer that selects minimal extra context to inspect."},
        {"role": "user", "content": selection_prompt},
    ],
    max_tokens=200,
    temperature=0.0,
)
sel_text = sel.choices[0].message.content
requested = []
try:
    import json
    if sel_text:
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
    "You are an expert code reviewer focusing on HIGH-FIDELITY issues only. "
    "Before flagging an issue, carefully examine the ENTIRE context including existing tests, documentation, and implementations. "
    "Only report issues that would cause actual problems: broken functionality, missing critical tests, security vulnerabilities, "
    "clear violations of stated requirements, poor design decisions, or incomplete implementation of requirements. "
    "Do NOT report issues if: (1) functionality already exists elsewhere in the codebase, "
    "(2) documentation already covers the topic adequately, (3) tests already provide sufficient coverage, "
    "or (4) the suggestion is purely cosmetic or incremental without clear impact. "
    "Your review must ensure: correctness, completeness (meets linked issue requirements), "
    "sound design (maintainable, aligned with project architecture), security/privacy compliance, and deterministic tests/CI. "
    "Output format: Each issue must include concrete evidence and impact. "
    "If no HIGH-FIDELITY issues exist, respond exactly: 'No actionable issues found.'"
)

prompt = f"""
Review this pull request for HIGH-FIDELITY issues only. 

CRITICAL: Before flagging any issue, search the ENTIRE codebase context below for existing implementations.

VALIDATION CHECKLIST (mark each as ✅ if already implemented or ❌ if missing):
1. Tests: Are there actually missing test cases for NEW functionality? (Don't flag if similar tests exist)
2. Documentation: Is there a genuine gap in docs that would confuse users? (Don't flag if adequately covered)
3. CI: Will this actually break CI or cause non-deterministic failures?
4. Security: Is there a real security vulnerability or credential exposure?
5. Functionality: Does this actually break working features or prevent the PR from meeting its goals?
6. Design: Does this violate project architecture or create maintainability issues?
7. Completeness: Does this fully address all requirements/acceptance criteria from the linked issue?

EVIDENCE REQUIRED: For each issue, provide:
- File path and specific line/function
- Concrete evidence the issue exists (quote relevant code)
- Clear impact: "This will cause X problem when Y happens"
- Verification it's not already handled elsewhere in the codebase

PRIORITY FILTERS:
- HIGH: Breaks functionality, security issues, missing critical tests for new features
- SKIP: Cosmetic improvements, redundant tests when coverage exists, documentation that's already adequate

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

Only report HIGH-FIDELITY issues with concrete evidence and impact. If none exist, reply: "No actionable issues found."
"""

review_resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ],
    max_tokens=800,  # Reduced to force conciseness and high-fidelity focus
    temperature=0.1,  # Lower temperature for more consistent, focused reviews
)
review = review_resp.choices[0].message.content

# Try to create a comment, but handle permission issues gracefully
# Skip commenting if no actionable issues were found
if review and "No actionable issues found" not in review.strip():
    try:
        pr.create_issue_comment(f"## Copilot LLM Review\n{review}")
        print("✅ Review comment posted successfully")
    except Exception as e:
        print(f"⚠️  Could not post comment (likely permissions): {e}")
        print("📝 Review content:")
        print("=" * 50)
        print(review)
        print("=" * 50)
else:
    print("✅ No actionable issues found - skipping comment")
