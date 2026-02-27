#!/usr/bin/env python3
"""Manage Claude Code sessions.

Usage:
    session-manager.py --list [--search <term>] [--json]
    session-manager.py --list-untitled [--json]
    session-manager.py --status [--json]
    session-manager.py --info <prefix> [--json]
    session-manager.py --messages <session-id> [--json] [--max-messages N]
    session-manager.py --themes [--json]
    session-manager.py --rebuild
    session-manager.py --smart [--search <term>]
    session-manager.py --smart <prefix>
    session-manager.py --auto-title --session-id <id>
    session-manager.py --delete <prefix>
    session-manager.py --purge [--dry-run]
    session-manager.py --plans [--search <term>] [--json]
    session-manager.py --plan-info <name> [--json]
    session-manager.py --plan-clean [--dry-run]
    session-manager.py --set-branch <branch>
    session-manager.py <prefix> <new-name>

Project detection (in order of priority):
    1. --project-dir <path>       Explicit sessions directory
    2. --transcript-path <path>   Derive from transcript path (hooks)
    3. CLAUDE_PROJECT_DIR env var  Compute sessions dir from project path
    4. Current working directory   Compute sessions dir from CWD

Examples:
    session-manager.py --list
    session-manager.py --list --search magento --json
    session-manager.py --status --json
    session-manager.py --info 03fb2b --json
    session-manager.py --messages 03fb2b --json
    session-manager.py --auto-title --session-id 03fb2b-...
    session-manager.py --purge --dry-run
    session-manager.py --smart
    session-manager.py 03fb2b "Rental boat overlap fix"
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ── Project detection ────────────────────────────────────────────────

PROJECT_DIR = None  # Set by detect_project_dir()
INDEX_FILE = None


def compute_sessions_dir(project_path):
    """Compute the Claude Code sessions directory from a project path.

    Claude Code stores sessions in ~/.claude/projects/<slug>/ where
    <slug> is the project path with '/' replaced by '-'.
    Example: /home/nacho/svn/addons160 -> -home-nacho-svn-addons160
    """
    slug = project_path.rstrip("/").replace("/", "-")
    return Path.home() / ".claude" / "projects" / slug


def detect_project_dir(opts):
    """Detect the Claude Code sessions directory."""
    # 1. Explicit --project-dir flag
    if opts.get("project_dir"):
        d = Path(opts["project_dir"])
        if d.exists():
            return d
        print(f"ERROR: --project-dir {d} does not exist", file=sys.stderr)
        sys.exit(1)

    # 2. Derive from transcript_path (e.g., from hooks)
    if opts.get("transcript_path"):
        d = Path(opts["transcript_path"]).parent
        if d.exists():
            return d

    # 3. CLAUDE_PROJECT_DIR env var
    env_dir = os.environ.get("CLAUDE_PROJECT_DIR")
    if env_dir:
        d = compute_sessions_dir(env_dir)
        if d.exists():
            return d

    # 4. Compute from CWD
    d = compute_sessions_dir(os.getcwd())
    if d.exists():
        return d

    # 5. If session-id provided, scan all project dirs
    session_id = opts.get("session_id")
    if session_id:
        projects_root = Path.home() / ".claude" / "projects"
        if projects_root.exists():
            for proj_dir in projects_root.iterdir():
                if proj_dir.is_dir() and (proj_dir / f"{session_id}.jsonl").exists():
                    return proj_dir

    # 6. List available projects
    projects_root = Path.home() / ".claude" / "projects"
    if projects_root.exists():
        available = sorted(
            d.name for d in projects_root.iterdir()
            if d.is_dir() and any(d.glob("*.jsonl"))
        )
        if len(available) == 1:
            return projects_root / available[0]
        if available:
            print("ERROR: Multiple projects found. Use --project-dir:", file=sys.stderr)
            for name in available:
                print(f"  {projects_root / name}", file=sys.stderr)
            sys.exit(1)

    print("ERROR: No Claude Code sessions directory found", file=sys.stderr)
    sys.exit(1)


# ── Index operations ─────────────────────────────────────────────────

def load_index():
    """Load sessions-index.json, creating it if missing."""
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            return json.load(f)
    return {"entries": []}


def save_index(data):
    with open(INDEX_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── JSONL parsing ────────────────────────────────────────────────────

def extract_user_messages(jsonl_path, max_messages=5, max_chars=300):
    """Extract first N real user messages from a JSONL session file."""
    messages = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                if len(messages) >= max_messages:
                    break
                entry = json.loads(line.strip())
                if entry.get("type") != "user":
                    continue
                content = entry.get("message", {}).get("content", "")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text = c["text"].strip()
                            if text.startswith("<") or \
                               text.startswith("The user opened") or \
                               text.startswith("The user selected"):
                                continue
                            clean = re.sub(r"<[^>]+>", "", text).strip()
                            if clean and len(clean) > 3:
                                messages.append(clean[:max_chars])
                                break
    except Exception:
        pass
    return messages


def scan_jsonl(jsonl_path, max_chars=80):
    """Extract title, first user message and message count from a JSONL."""
    custom_title = ""
    summary = ""
    first_msg = ""
    msg_count = 0
    try:
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                entry_type = entry.get("type", "")
                if entry_type == "custom-title" and not custom_title:
                    custom_title = entry.get("customTitle", "")
                elif entry_type == "summary" and not summary:
                    summary = entry.get("summary", "")
                elif entry_type in ("user", "assistant"):
                    msg_count += 1
                    if not first_msg and entry_type == "user":
                        content = entry.get("message", {}).get("content", "")
                        if isinstance(content, list):
                            for c in content:
                                if not isinstance(c, dict) or c.get("type") != "text":
                                    continue
                                text = c["text"].strip()
                                if text.startswith("<") or \
                                   text.startswith("The user opened") or \
                                   text.startswith("The user selected"):
                                    continue
                                clean = re.sub(r"<[^>]+>", "", text).strip()
                                if clean and len(clean) > 3:
                                    first_msg = clean[:max_chars]
                                    break
    except Exception:
        pass
    title = custom_title or summary
    has_custom_title = bool(custom_title)
    return title, first_msg, msg_count, has_custom_title


def scan_jsonl_full(jsonl_path):
    """Full scan of a JSONL to extract all index-relevant metadata."""
    custom_title = ""
    summary = ""
    first_prompt = ""
    first_msg_text = ""
    git_branch = ""
    project_path = ""
    is_sidechain = False
    created = ""
    modified = ""
    msg_count = 0
    session_id = jsonl_path.stem

    try:
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                entry_type = entry.get("type", "")

                if entry_type == "custom-title" and not custom_title:
                    custom_title = entry.get("customTitle", "")

                elif entry_type == "summary" and not summary:
                    summary = entry.get("summary", "")

                elif entry_type in ("user", "assistant"):
                    msg_count += 1
                    ts = entry.get("timestamp", "")

                    if not created and ts:
                        created = ts if isinstance(ts, str) else ""
                    if ts:
                        modified = ts if isinstance(ts, str) else ""

                    if not git_branch:
                        git_branch = entry.get("gitBranch", "")
                    if not project_path:
                        project_path = entry.get("cwd", "")
                    if entry.get("isSidechain"):
                        is_sidechain = True

                    if entry_type == "user":
                        content = entry.get("message", {}).get("content", "")
                        raw = ""
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "text":
                                    text = c["text"]
                                    if text.strip().startswith("<ide_") or \
                                       text.strip().startswith("The user opened") or \
                                       text.strip().startswith("The user selected"):
                                        continue
                                    raw = text
                                    break
                        elif isinstance(content, str):
                            raw = content
                        if raw and not first_msg_text:
                            if not first_prompt:
                                first_prompt = raw[:200]
                            clean = re.sub(r"<[^>]+>", "", raw).strip()
                            if clean and len(clean) > 3 and not clean.startswith("<"):
                                first_msg_text = clean[:80]
    except Exception:
        pass

    mtime = jsonl_path.stat().st_mtime
    return {
        "sessionId": session_id,
        "fullPath": str(jsonl_path),
        "fileMtime": int(mtime * 1000),
        "firstPrompt": first_prompt or first_msg_text,
        "summary": custom_title or summary,
        "hasCustomTitle": bool(custom_title),
        "messageCount": msg_count,
        "created": created or datetime.fromtimestamp(mtime).isoformat() + "Z",
        "modified": modified or created or datetime.fromtimestamp(mtime).isoformat() + "Z",
        "gitBranch": git_branch,
        "projectPath": project_path,
        "isSidechain": is_sidechain,
    }


def parse_date(iso_str):
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


# ── Session operations ───────────────────────────────────────────────

def get_all_sessions():
    """Get all sessions from index + unindexed JSONL files."""
    data = load_index()
    indexed = {e["sessionId"] for e in data.get("entries", [])}
    entries = list(data.get("entries", []))

    for f in PROJECT_DIR.glob("*.jsonl"):
        sid = f.stem
        if sid not in indexed and sid != "sessions-index":
            mtime = f.stat().st_mtime
            title, first_msg, msg_count, has_ct = scan_jsonl(f)
            entries.append({
                "sessionId": sid,
                "summary": title,
                "firstPrompt": first_msg,
                "created": datetime.fromtimestamp(mtime).isoformat() + "Z",
                "messageCount": msg_count,
                "hasCustomTitle": has_ct,
            })

    entries.sort(key=lambda e: parse_date(e.get("created", "")), reverse=True)
    return entries


def filter_sessions(entries, search=None):
    """Filter sessions by search term."""
    if not search:
        return entries
    term = search.lower()
    return [
        e for e in entries
        if term in e.get("summary", "").lower()
        or term in e.get("firstPrompt", "").lower()
        or term in e.get("sessionId", "").lower()
    ]


# ── Theme classification ─────────────────────────────────────────────

_theme_rules = None  # Lazy-loaded from session-themes.json


def get_themes_file():
    """Get themes file path (requires PROJECT_DIR to be set)."""
    if PROJECT_DIR:
        return PROJECT_DIR / "session-themes.json"
    return None


def load_theme_rules():
    """Load theme rules from session-themes.json. Returns list of (name, keywords)."""
    global _theme_rules
    if _theme_rules is not None:
        return _theme_rules

    _theme_rules = []
    themes_file = get_themes_file()
    if themes_file and themes_file.exists():
        try:
            with open(themes_file) as f:
                data = json.load(f)
            for theme in data.get("themes", []):
                name = theme.get("name", "")
                keywords = theme.get("keywords", [])
                if name and keywords:
                    _theme_rules.append((name, keywords))
        except Exception:
            pass
    return _theme_rules


def save_theme_rules(rules):
    """Save theme rules to session-themes.json."""
    global _theme_rules
    _theme_rules = rules
    themes_file = get_themes_file()
    if not themes_file:
        return
    data = {
        "themes": [{"name": name, "keywords": kws} for name, kws in rules],
        "lastUpdated": datetime.now().isoformat()[:10],
    }
    with open(themes_file, "w") as f:
        json.dump(data, f, indent=2)


def classify_theme(entry):
    """Classify a session into a theme based on summary, firstPrompt and branch."""
    rules = load_theme_rules()
    if not rules:
        return ""
    text = " ".join([
        entry.get("summary", ""),
        entry.get("firstPrompt", ""),
        entry.get("gitBranch", ""),
    ]).lower()
    for theme, keywords in rules:
        for kw in keywords:
            if kw in text:
                return theme
    return ""


def generate_theme_rules():
    """Auto-generate theme rules from existing session titles using Claude."""
    entries = get_all_sessions()
    titled = [e for e in entries if e.get("hasCustomTitle") and e.get("summary")]
    if not titled:
        print("No titled sessions to analyze.", file=sys.stderr)
        return []

    # Collect all titles
    titles = [e["summary"] for e in titled[:100]]  # Cap at 100
    titles_text = "\n".join(f"- {t}" for t in titles)

    prompt = (
        "Analiza estos titulos de sesiones de desarrollo y genera un mapa tematico.\n"
        "Agrupa los titulos en categorias (maximo 15) con keywords para clasificar.\n"
        "Responde SOLO con JSON valido, sin markdown ni explicaciones:\n"
        '[{"name":"NombreCategoria","keywords":["keyword1","keyword2"]},...]\n\n'
        "Las keywords deben ser fragmentos de texto en minusculas que aparecen en los titulos.\n"
        "Incluye keywords tanto en español como en ingles si es relevante.\n\n"
        f"Titulos:\n{titles_text}"
    )

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "haiku", prompt],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            # Try to extract JSON from response
            match = re.search(r'\[.*\]', output, re.DOTALL)
            if match:
                themes = json.loads(match.group())
                rules = []
                for t in themes:
                    name = t.get("name", "")
                    kws = t.get("keywords", [])
                    if name and kws:
                        rules.append((name, [k.lower() for k in kws]))
                return rules
    except Exception as e:
        print(f"Error generating themes: {e}", file=sys.stderr)

    return []


def cmd_themes(as_json=False):
    """Generate or show theme rules."""
    rules = load_theme_rules()

    if rules:
        if as_json:
            print(json.dumps([{"name": n, "keywords": k} for n, k in rules], indent=2))
        else:
            print(f"Theme map ({len(rules)} themes):\n")
            for name, keywords in rules:
                print(f"  [{name}] {', '.join(keywords[:5])}" +
                      (f" (+{len(keywords)-5})" if len(keywords) > 5 else ""))
            themes_file = get_themes_file()
            print(f"\nStored in: {themes_file}")
    else:
        print("No theme map found. Generating from session titles...")
        rules = generate_theme_rules()
        if rules:
            save_theme_rules(rules)
            print(f"\nGenerated {len(rules)} themes:")
            for name, keywords in rules:
                print(f"  [{name}] {', '.join(keywords[:5])}")
            themes_file = get_themes_file()
            print(f"\nSaved to: {themes_file}")
        else:
            print("Could not generate themes (no titled sessions or Claude unavailable).")


# ── Commands ─────────────────────────────────────────────────────────

def has_custom_title(session_id):
    """Check if a session JSONL has a custom-title entry."""
    jsonl_path = PROJECT_DIR / f"{session_id}.jsonl"
    if not jsonl_path.exists():
        return False
    try:
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get("type") == "custom-title":
                    return True
    except Exception:
        pass
    return False


def find_session(prefix):
    """Find a single session matching a prefix."""
    matches = set()
    for f in PROJECT_DIR.glob(f"{prefix}*.jsonl"):
        matches.add(f.stem)

    data = load_index()
    for e in data.get("entries", []):
        if e["sessionId"].startswith(prefix):
            matches.add(e["sessionId"])

    matches = sorted(matches)

    if not matches:
        print(f'ERROR: No session found matching "{prefix}"', file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        print(f'ERROR: {len(matches)} sessions match "{prefix}":', file=sys.stderr)
        for m in matches:
            print(f"  {m}", file=sys.stderr)
        print("Use a longer prefix.", file=sys.stderr)
        sys.exit(1)

    return matches[0]


def apply_custom_title(session_id, new_name):
    """Write custom-title to both JSONL and sessions-index.json."""
    # Update index
    data = load_index()
    for e in data.get("entries", []):
        if e["sessionId"] == session_id:
            e["summary"] = new_name
            e["hasCustomTitle"] = True
            break
    save_index(data)

    # Update JSONL (preserving original mtime)
    jsonl_path = PROJECT_DIR / f"{session_id}.jsonl"
    if not jsonl_path.exists():
        return

    original_stat = jsonl_path.stat()
    original_times = (original_stat.st_atime, original_stat.st_mtime)

    # VSCode reads only the last 64KB of the JSONL to find customTitle.
    # Strategy: update existing custom-title entries AND always append one
    # at the end so it's always within the 64KB tail window.
    lines = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("type") == "custom-title":
                    entry["customTitle"] = new_name
                    lines.append(json.dumps(entry) + "\n")
                    continue
            except Exception:
                pass
            lines.append(line)

    # Always append at the end to ensure it's in the last 64KB
    lines.append(json.dumps({
        "type": "custom-title",
        "customTitle": new_name,
        "sessionId": session_id,
    }) + "\n")

    with open(jsonl_path, "w") as f:
        f.writelines(lines)

    os.utime(jsonl_path, original_times)


def generate_smart_title(session_id):
    """Use Claude CLI to generate a short descriptive title for a session."""
    jsonl_path = PROJECT_DIR / f"{session_id}.jsonl"
    if not jsonl_path.exists():
        return None

    messages = extract_user_messages(jsonl_path, max_messages=5, max_chars=300)
    if not messages:
        return None

    msgs_text = "\n---\n".join(messages)
    prompt = (
        "Estos son los primeros mensajes del usuario en una sesion de desarrollo Odoo. "
        "Genera un titulo corto (maximo 60 caracteres) en castellano que resuma el tema principal. "
        "Solo responde con el titulo, sin comillas ni explicaciones.\n\n"
        f"{msgs_text}"
    )

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "haiku", prompt],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            title = result.stdout.strip()
            title = title.strip('"\'')
            if len(title) > 70:
                title = title[:67] + "..."
            return title
    except Exception as e:
        print(f"  Error calling claude: {e}", file=sys.stderr)

    return None


# ── Command implementations ──────────────────────────────────────────

def cmd_list(search=None, as_json=False):
    """List sessions, optionally filtered by search term."""
    entries = filter_sessions(get_all_sessions(), search)

    if as_json:
        result = []
        for e in entries:
            result.append({
                "sessionId": e["sessionId"],
                "created": e.get("created", "")[:10],
                "messageCount": e.get("messageCount", 0),
                "summary": e.get("summary", ""),
                "firstPrompt": e.get("firstPrompt", ""),
                "hasCustomTitle": e.get("hasCustomTitle", False),
                "theme": classify_theme(e),
            })
        print(json.dumps(result, indent=2))
        return

    if search:
        print(f'Filtro: "{search}" ({len(entries)} resultados)\n')

    for e in entries:
        msgs = e.get("messageCount", 0)
        sid = e["sessionId"]
        summary = e.get("summary", "")
        first_raw = e.get("firstPrompt", "")
        first = re.sub(r"<[^>]*>", "", first_raw).strip()
        if first.startswith("The user opened") or first.startswith("The user selected"):
            first = ""
        first = re.sub(r"/\S{20,}", "...", first)
        first = first[:60]
        created = e.get("created", "")[:10]
        display = summary if summary else first if first else "(sin titulo)"
        has_ct = e.get("hasCustomTitle", False)
        if not has_ct and "hasCustomTitle" not in e:
            has_ct = has_custom_title(e["sessionId"]) if (PROJECT_DIR / f"{e['sessionId']}.jsonl").exists() else False
        named = "*" if has_ct else " "
        theme = classify_theme(e)
        theme_tag = f"[{theme:11s}]" if theme else " " * 13
        print(f"{sid[:8]}  {created}  {msgs:>4} msgs {named} {theme_tag} {display[:60]}")


def cmd_list_untitled(as_json=False):
    """List sessions without custom titles that have >3 messages."""
    entries = get_all_sessions()
    untitled = []
    for e in entries:
        if e.get("messageCount", 0) <= 3:
            continue
        if e.get("hasCustomTitle"):
            continue
        if "hasCustomTitle" not in e and has_custom_title(e["sessionId"]):
            continue
        untitled.append(e)

    if as_json:
        result = []
        for e in untitled:
            result.append({
                "sessionId": e["sessionId"],
                "messageCount": e.get("messageCount", 0),
                "firstPrompt": e.get("firstPrompt", ""),
                "created": e.get("created", "")[:10],
            })
        print(json.dumps(result, indent=2))
    else:
        if not untitled:
            print("All sessions have custom titles.")
            return
        print(f"{len(untitled)} untitled sessions:\n")
        for e in untitled:
            sid = e["sessionId"]
            msgs = e.get("messageCount", 0)
            first = re.sub(r"<[^>]+>", "", e.get("firstPrompt", "")).strip()[:50]
            print(f"  {sid[:8]} ({msgs:>4} msgs) {first}")


def cmd_status(as_json=False):
    """Show summary statistics about sessions."""
    entries = get_all_sessions()
    total = len(entries)
    untitled = 0
    empty = 0
    for e in entries:
        mc = e.get("messageCount", 0)
        if mc <= 3:
            empty += 1
        elif not e.get("hasCustomTitle"):
            if "hasCustomTitle" not in e and has_custom_title(e["sessionId"]):
                continue
            untitled += 1

    if as_json:
        print(json.dumps({"total": total, "untitled": untitled, "empty": empty}))
    else:
        print(f"Total: {total}  Untitled: {untitled}  Empty: {empty}")


def cmd_info(prefix, as_json=False):
    """Show detailed info about a specific session."""
    session_id = find_session(prefix)
    jsonl_path = PROJECT_DIR / f"{session_id}.jsonl"

    if not jsonl_path.exists():
        if as_json:
            print(json.dumps({"error": "JSONL file not found", "sessionId": session_id}))
        else:
            print(f"ERROR: JSONL file not found for {session_id}")
        sys.exit(1)

    info = scan_jsonl_full(jsonl_path)
    size_kb = jsonl_path.stat().st_size / 1024

    if as_json:
        info["sizeKb"] = round(size_kb, 1)
        info["theme"] = classify_theme(info)
        print(json.dumps(info, indent=2))
    else:
        display = info["summary"] or info["firstPrompt"] or "(sin titulo)"
        print(f"Session:  {session_id}")
        print(f"Title:    {display}")
        print(f"Created:  {info['created'][:19]}")
        print(f"Modified: {info['modified'][:19]}")
        print(f"Messages: {info['messageCount']}")
        print(f"Size:     {size_kb:.1f} KB")
        print(f"Branch:   {info['gitBranch']}")
        print(f"Custom:   {'Yes' if info['hasCustomTitle'] else 'No'}")
        theme = classify_theme(info)
        if theme:
            print(f"Theme:    {theme}")


def cmd_messages(session_id_or_prefix, as_json=False, max_messages=5):
    """Extract user messages from a session."""
    session_id = find_session(session_id_or_prefix)
    jsonl_path = PROJECT_DIR / f"{session_id}.jsonl"

    if not jsonl_path.exists():
        if as_json:
            print(json.dumps({"error": "JSONL file not found"}))
        else:
            print(f"ERROR: JSONL file not found for {session_id}")
        sys.exit(1)

    messages = extract_user_messages(jsonl_path, max_messages=max_messages, max_chars=500)

    if as_json:
        print(json.dumps({
            "sessionId": session_id,
            "messages": [{"role": "user", "content": m} for m in messages],
        }, indent=2))
    else:
        for i, m in enumerate(messages, 1):
            print(f"[{i}] {m[:200]}")


def cmd_rebuild():
    """Rebuild sessions-index.json from all JSONL files on disk."""
    data = load_index()
    indexed = {e["sessionId"]: e for e in data.get("entries", [])}

    jsonl_files = [
        f for f in PROJECT_DIR.glob("*.jsonl")
        if f.stem != "sessions-index"
    ]

    added = 0
    updated = 0
    for jsonl_path in jsonl_files:
        sid = jsonl_path.stem
        entry = scan_jsonl_full(jsonl_path)
        if entry["messageCount"] == 0 and sid not in indexed:
            continue
        if sid in indexed:
            old = indexed[sid]
            had_ct = old.get("hasCustomTitle", False)
            if had_ct and not entry.get("hasCustomTitle"):
                entry["summary"] = old["summary"]
                entry["hasCustomTitle"] = True
            indexed[sid] = entry
            updated += 1
        else:
            indexed[sid] = entry
            added += 1

    on_disk = {f.stem for f in jsonl_files}
    orphaned = [sid for sid in indexed if sid not in on_disk]
    for sid in orphaned:
        del indexed[sid]

    entries = list(indexed.values())
    entries.sort(key=lambda e: parse_date(e.get("created", "")), reverse=True)
    data["entries"] = entries
    save_index(data)

    total = len(data["entries"])
    parts = [f"{total} sessions", f"{added} added", f"{updated} updated"]
    if orphaned:
        parts.append(f"{len(orphaned)} orphaned removed")
    parts.append(f"{len(jsonl_files)} on disk")
    print(f"Rebuild complete: {', '.join(parts)}")


def cmd_rename(prefix, new_name):
    """Rename a session by prefix."""
    session_id = find_session(prefix)
    print(f"Session: {session_id}")
    apply_custom_title(session_id, new_name)
    print(f'Done: "{new_name}"')


def cmd_smart(prefix=None, search=None):
    """Smart rename: one session or all without custom title."""
    if prefix:
        session_id = find_session(prefix)
        if has_custom_title(session_id):
            jsonl_path = PROJECT_DIR / f"{session_id}.jsonl"
            title, _, _, _ = scan_jsonl(jsonl_path)
            print(f'{session_id[:8]}: already named "{title}"')
            return

        print(f"{session_id[:8]}: generating title...", end=" ", flush=True)
        new_title = generate_smart_title(session_id)
        if new_title:
            apply_custom_title(session_id, new_title)
            print(f'-> "{new_title}"')
        else:
            print("-> (could not generate)")
        return

    entries = get_all_sessions()
    if search:
        entries = filter_sessions(entries, search)

    to_rename = []
    for e in entries:
        if e.get("messageCount", 0) <= 2:
            continue
        if e.get("hasCustomTitle"):
            continue
        if "hasCustomTitle" not in e and has_custom_title(e["sessionId"]):
            continue
        to_rename.append(e)

    if not to_rename:
        msg = "All sessions already have custom titles"
        if search:
            msg += f' (filtered by "{search}")'
        print(msg)
        return

    print(f"Smart renaming {len(to_rename)} sessions...\n")

    for e in to_rename:
        sid = e["sessionId"]
        msgs = e.get("messageCount", 0)
        first = re.sub(r"<[^>]+>", "", e.get("firstPrompt", "")).strip()[:50]
        print(f"{sid[:8]} ({msgs:>4} msgs) {first[:40]}...", end=" ", flush=True)

        new_title = generate_smart_title(sid)
        if new_title:
            apply_custom_title(sid, new_title)
            print(f'-> "{new_title}"')
        else:
            print("-> (skip)")

    print("\nDone.")


def cmd_auto_title(session_id):
    """Auto-title a session (for SessionEnd hook). Silent on success."""
    jsonl_path = PROJECT_DIR / f"{session_id}.jsonl"
    if not jsonl_path.exists():
        return

    # Skip if already has custom title
    if has_custom_title(session_id):
        return

    # Skip if too few messages
    _, _, msg_count, _ = scan_jsonl(jsonl_path)
    if msg_count <= 3:
        return

    new_title = generate_smart_title(session_id)
    if new_title:
        apply_custom_title(session_id, new_title)


def cmd_delete(prefix):
    """Delete a specific session."""
    session_id = find_session(prefix)
    jsonl_path = PROJECT_DIR / f"{session_id}.jsonl"

    if jsonl_path.exists():
        title, first_msg, msg_count, _ = scan_jsonl(jsonl_path)
        display = title or first_msg or "(sin titulo)"
        print(f"Deleting: {session_id[:8]} ({msg_count} msgs) {display[:60]}")
        jsonl_path.unlink()
    else:
        print(f"Deleting: {session_id[:8]} (no JSONL on disk)")

    data = load_index()
    data["entries"] = [e for e in data.get("entries", []) if e["sessionId"] != session_id]
    save_index(data)
    print("Done.")


def cmd_purge(dry_run=False):
    """Delete all sessions with no real content (<=3 messages)."""
    to_delete = []

    for jsonl_path in sorted(PROJECT_DIR.glob("*.jsonl")):
        if jsonl_path.stem == "sessions-index":
            continue
        sid = jsonl_path.stem
        _, _, msg_count, _ = scan_jsonl(jsonl_path)
        if msg_count <= 3:
            size_kb = jsonl_path.stat().st_size / 1024
            to_delete.append((sid, jsonl_path, size_kb, msg_count))

    if not to_delete:
        print("No empty sessions to purge.")
        return

    for sid, path, size_kb, mc in to_delete:
        action = "would delete" if dry_run else "deleted"
        print(f"  {sid[:8]} ({size_kb:.0f}KB, {mc} msgs) -> {action}")

    if dry_run:
        print(f"\nDry run: {len(to_delete)} sessions would be purged.")
        return

    for _, path, _, _ in to_delete:
        path.unlink()

    data = load_index()
    on_disk = {f.stem for f in PROJECT_DIR.glob("*.jsonl") if f.stem != "sessions-index"}
    before = len(data.get("entries", []))
    data["entries"] = [e for e in data.get("entries", []) if e["sessionId"] in on_disk]
    after = len(data["entries"])
    save_index(data)

    kept = len([f for f in PROJECT_DIR.glob("*.jsonl") if f.stem != "sessions-index"])
    print(f"\nPurged {len(to_delete)} empty sessions, kept {kept}. Index: {before} -> {after} entries.")


# ── Plan operations ──────────────────────────────────────────────────

PLANS_DIR = Path.home() / ".claude" / "plans"


def get_plan_title(plan_path):
    """Extract title from a plan markdown file."""
    try:
        with open(plan_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("# Plan: "):
                    return line[8:]
                if line.startswith("# "):
                    return line[2:]
    except Exception:
        pass
    return ""


def _verify_plan_link(jsonl_path, plan_filename, plan_stem):
    """Verify a plan link by parsing JSONL entries structurally.

    Returns True if the session contains a real plan interaction:
    - An entry with slug == plan_stem (session created for this plan)
    - A tool_use (Read/Write/Edit) with file_path pointing to the plan
    """
    plan_path_suffix = "/.claude/plans/" + plan_filename
    with open(jsonl_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            # Check 1: slug field at entry level
            if entry.get("slug") == plan_stem:
                return True

            # Check 2: tool_use with file_path pointing to the plan
            msg = entry.get("message")
            if not msg or not isinstance(msg, dict):
                continue
            for block in msg.get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue
                inp = block.get("input")
                if not isinstance(inp, dict):
                    continue
                fp = inp.get("file_path") or inp.get("path") or ""
                if fp.endswith(plan_path_suffix):
                    return True
    return False


def find_plan_sessions(plan_filename):
    """Find sessions that reference a plan file via tool calls or slug.

    Two-phase approach for speed + precision:
    1. grep pre-filter: quickly narrows 80+ JSONL files to a few candidates
    2. Structured JSON verification: parses only candidates to confirm
       real interactions (slug match or tool_use with file_path)

    This avoids false positives from listing plans, IDE selections, or
    casual mentions of the plan filename in conversation.
    """
    sessions = []
    current_sid = os.environ.get("CLAUDE_SESSION_ID", "")
    jsonl_files = [
        f for f in PROJECT_DIR.glob("*.jsonl")
        if f.stem != "sessions-index" and f.stem != current_sid
    ]

    plan_stem = Path(plan_filename).stem

    if not jsonl_files:
        return sessions

    # Phase 1: grep pre-filter (fast — narrows to candidates)
    candidates = []
    try:
        result = subprocess.run(
            ["grep", "-l", plan_filename] + [str(f) for f in jsonl_files],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    candidates.append(Path(line))
    except Exception:
        # If grep fails, all files are candidates
        candidates = list(jsonl_files)

    # Phase 2: structured JSON verification (precise)
    for jsonl_path in candidates:
        try:
            if _verify_plan_link(jsonl_path, plan_filename, plan_stem):
                sessions.append(jsonl_path.stem)
        except Exception:
            pass

    return sessions


def get_session_title(session_id):
    """Get session title from the index."""
    data = load_index()
    for e in data.get("entries", []):
        if e["sessionId"] == session_id:
            return e.get("summary", "") or e.get("firstPrompt", "")[:60]
    return ""


def cmd_plans(search=None, as_json=False):
    """List all plan files with metadata and linked sessions."""
    if not PLANS_DIR.exists():
        if as_json:
            print("[]")
        else:
            print("No plans directory found.")
        return

    plans = sorted(PLANS_DIR.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not plans:
        if as_json:
            print("[]")
        else:
            print("No plans found.")
        return

    results = []
    for plan_path in plans:
        title = get_plan_title(plan_path)
        name = plan_path.stem
        stat = plan_path.stat()
        size_kb = stat.st_size / 1024
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")
        linked = find_plan_sessions(plan_path.name)

        # Build session info
        session_info = []
        for sid in linked:
            stitle = get_session_title(sid)
            session_info.append({"sessionId": sid, "title": stitle})

        results.append({
            "name": name,
            "filename": plan_path.name,
            "title": title,
            "modified": modified,
            "sizeKb": round(size_kb, 1),
            "sessions": session_info,
        })

    # Filter by search term
    if search:
        term = search.lower()
        results = [
            r for r in results
            if term in r["name"].lower()
            or term in r["title"].lower()
            or any(term in s["title"].lower() for s in r["sessions"])
        ]

    if as_json:
        print(json.dumps(results, indent=2))
        return

    if search:
        print(f'Filtro: "{search}" ({len(results)} resultados)\n')

    for r in results:
        scount = len(r["sessions"])
        sess_str = ", ".join(s["sessionId"][:8] for s in r["sessions"]) if scount else "-"
        title_display = r["title"][:45] if r["title"] else "(sin titulo)"
        print(f"  {r['name'][:30]:<30s}  {r['modified']}  {r['sizeKb']:>6.1f}KB  {scount} sess  {title_display}")
        if scount and not search:
            for s in r["sessions"][:3]:
                stitle = s["title"][:50] if s["title"] else "(sin titulo)"
                print(f"    -> {s['sessionId'][:8]} {stitle}")
            if scount > 3:
                print(f"    ... +{scount - 3} mas")

    print(f"\nTotal: {len(results)} planes")


def cmd_plan_info(name, as_json=False):
    """Show detailed info about a specific plan."""
    # Find matching plan
    matches = list(PLANS_DIR.glob(f"{name}*.md"))
    if not matches:
        print(f'ERROR: No plan found matching "{name}"', file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        print(f'ERROR: {len(matches)} plans match "{name}":', file=sys.stderr)
        for m in matches:
            print(f"  {m.stem}", file=sys.stderr)
        sys.exit(1)

    plan_path = matches[0]
    title = get_plan_title(plan_path)
    stat = plan_path.stat()
    size_kb = stat.st_size / 1024
    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
    linked = find_plan_sessions(plan_path.name)

    # Read preview
    preview = ""
    try:
        with open(plan_path) as f:
            preview = f.read(500)
    except Exception:
        pass

    session_info = []
    for sid in linked:
        stitle = get_session_title(sid)
        session_info.append({"sessionId": sid, "title": stitle})

    if as_json:
        print(json.dumps({
            "name": plan_path.stem,
            "filename": plan_path.name,
            "title": title,
            "modified": modified,
            "sizeKb": round(size_kb, 1),
            "sessions": session_info,
            "preview": preview,
            "fullPath": str(plan_path),
        }, indent=2))
        return

    print(f"Plan:     {plan_path.stem}")
    print(f"Title:    {title or '(sin titulo)'}")
    print(f"Modified: {modified}")
    print(f"Size:     {size_kb:.1f} KB")
    print(f"Path:     {plan_path}")
    print(f"Sessions: {len(linked)}")
    for s in session_info:
        stitle = s["title"][:50] if s["title"] else "(sin titulo)"
        print(f"  -> {s['sessionId'][:8]} {stitle}")
    if preview:
        print(f"\n--- Preview ---")
        print(preview)
        if len(preview) >= 500:
            print("...")


def cmd_plan_clean(dry_run=False):
    """Clean up orphaned plans (no linked sessions)."""
    if not PLANS_DIR.exists():
        print("No plans directory found.")
        return

    plans = list(PLANS_DIR.glob("*.md"))
    if not plans:
        print("No plans found.")
        return

    orphans = []
    for plan_path in sorted(plans):
        linked = find_plan_sessions(plan_path.name)
        if not linked:
            title = get_plan_title(plan_path)
            size_kb = plan_path.stat().st_size / 1024
            orphans.append((plan_path, title, size_kb))

    if not orphans:
        print(f"No orphaned plans found ({len(plans)} plans, all linked to sessions).")
        return

    for path, title, size_kb in orphans:
        action = "would move to .trash" if dry_run else "moved to .trash"
        display = title[:40] if title else "(sin titulo)"
        print(f"  {path.stem[:30]:<30s}  {size_kb:>5.1f}KB  {display}  -> {action}")

    if dry_run:
        print(f"\nDry run: {len(orphans)} orphaned plans would be cleaned.")
        return

    trash_dir = PLANS_DIR / ".trash"
    trash_dir.mkdir(exist_ok=True)

    for path, _, _ in orphans:
        dest = trash_dir / path.name
        path.rename(dest)

    remaining = len(list(PLANS_DIR.glob("*.md")))
    print(f"\nMoved {len(orphans)} orphaned plans to .trash/. Remaining: {remaining} plans.")


def cmd_set_branch(new_branch):
    """Change gitBranch on ALL sessions."""
    data = load_index()
    index_count = 0
    for e in data.get("entries", []):
        if e.get("gitBranch") != new_branch:
            e["gitBranch"] = new_branch
            index_count += 1
    save_index(data)

    file_count = 0
    for jsonl_path in sorted(PROJECT_DIR.glob("*.jsonl")):
        if jsonl_path.stem == "sessions-index":
            continue

        original_stat = jsonl_path.stat()
        original_times = (original_stat.st_atime, original_stat.st_mtime)

        lines = []
        changed = False
        with open(jsonl_path) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("type") in ("user", "assistant") and \
                       "gitBranch" in entry and entry["gitBranch"] != new_branch:
                        entry["gitBranch"] = new_branch
                        lines.append(json.dumps(entry) + "\n")
                        changed = True
                        continue
                except Exception:
                    pass
                lines.append(line)

        if changed:
            with open(jsonl_path, "w") as f:
                f.writelines(lines)
            os.utime(jsonl_path, original_times)
            file_count += 1

    print(f"Branch set to '{new_branch}': {index_count} index entries, {file_count} JSONL files updated.")


# ── Argument parsing ─────────────────────────────────────────────────

def parse_args():
    args = sys.argv[1:]
    opts = {
        "search": None,
        "smart": False,
        "rebuild": False,
        "list": False,
        "list_untitled": False,
        "status": False,
        "info": False,
        "messages": None,
        "help": False,
        "delete": False,
        "purge": False,
        "dry_run": False,
        "auto_title": False,
        "themes": False,
        "plans": False,
        "plan_info": False,
        "plan_clean": False,
        "session_id": None,
        "set_branch": None,
        "project_dir": None,
        "transcript_path": None,
        "json_output": False,
        "max_messages": 5,
        "positional": [],
    }

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-h", "--help"):
            opts["help"] = True
        elif arg in ("-l", "--list"):
            opts["list"] = True
        elif arg == "--list-untitled":
            opts["list_untitled"] = True
        elif arg == "--status":
            opts["status"] = True
        elif arg == "--info":
            opts["info"] = True
        elif arg == "--messages" and i + 1 < len(args):
            i += 1
            opts["messages"] = args[i]
        elif arg in ("-r", "--rebuild"):
            opts["rebuild"] = True
        elif arg in ("-s", "--smart"):
            opts["smart"] = True
        elif arg in ("-d", "--delete"):
            opts["delete"] = True
        elif arg == "--purge":
            opts["purge"] = True
        elif arg == "--dry-run":
            opts["dry_run"] = True
        elif arg == "--auto-title":
            opts["auto_title"] = True
        elif arg == "--themes":
            opts["themes"] = True
        elif arg == "--plans":
            opts["plans"] = True
        elif arg == "--plan-info":
            opts["plan_info"] = True
        elif arg == "--plan-clean":
            opts["plan_clean"] = True
        elif arg == "--session-id" and i + 1 < len(args):
            i += 1
            opts["session_id"] = args[i]
        elif arg == "--set-branch" and i + 1 < len(args):
            i += 1
            opts["set_branch"] = args[i]
        elif arg == "--project-dir" and i + 1 < len(args):
            i += 1
            opts["project_dir"] = args[i]
        elif arg == "--transcript-path" and i + 1 < len(args):
            i += 1
            opts["transcript_path"] = args[i]
        elif arg == "--search" and i + 1 < len(args):
            i += 1
            opts["search"] = args[i]
        elif arg == "--json":
            opts["json_output"] = True
        elif arg == "--max-messages" and i + 1 < len(args):
            i += 1
            opts["max_messages"] = int(args[i])
        else:
            opts["positional"].append(arg)
        i += 1

    return opts


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    opts = parse_args()

    has_action = (
        opts["smart"] or opts["rebuild"] or opts["list"] or opts["list_untitled"]
        or opts["status"] or opts["info"] or opts["messages"] or opts["delete"]
        or opts["purge"] or opts["set_branch"] or opts["auto_title"] or opts["themes"]
        or opts["plans"] or opts["plan_info"] or opts["plan_clean"]
        or opts["positional"]
    )

    if opts["help"] or not has_action:
        print(__doc__)
        sys.exit(0)

    # Detect project directory
    PROJECT_DIR = detect_project_dir(opts)
    INDEX_FILE = PROJECT_DIR / "sessions-index.json"

    # Auto-rebuild index if missing
    if not INDEX_FILE.exists():
        jsonl_count = len(list(PROJECT_DIR.glob("*.jsonl")))
        if jsonl_count > 0:
            print(f"Index not found, rebuilding from {jsonl_count} JSONL files...",
                  file=sys.stderr)
            save_index({"entries": []})
            cmd_rebuild()
        else:
            save_index({"entries": []})

    # Dispatch
    j = opts["json_output"]

    if opts["plans"]:
        cmd_plans(search=opts["search"], as_json=j)
    elif opts["plan_info"]:
        if not opts["positional"]:
            print("ERROR: --plan-info requires a plan name", file=sys.stderr)
            sys.exit(1)
        cmd_plan_info(opts["positional"][0], as_json=j)
    elif opts["plan_clean"]:
        cmd_plan_clean(dry_run=opts["dry_run"])
    elif opts["themes"]:
        cmd_themes(as_json=j)
    elif opts["rebuild"]:
        cmd_rebuild()
    elif opts["set_branch"]:
        cmd_set_branch(opts["set_branch"])
    elif opts["auto_title"]:
        if not opts["session_id"]:
            print("ERROR: --auto-title requires --session-id", file=sys.stderr)
            sys.exit(1)
        cmd_auto_title(opts["session_id"])
    elif opts["status"]:
        cmd_status(as_json=j)
    elif opts["list_untitled"]:
        cmd_list_untitled(as_json=j)
    elif opts["info"]:
        if not opts["positional"]:
            print("ERROR: --info requires a session prefix", file=sys.stderr)
            sys.exit(1)
        cmd_info(opts["positional"][0], as_json=j)
    elif opts["messages"]:
        cmd_messages(opts["messages"], as_json=j, max_messages=opts["max_messages"])
    elif opts["purge"]:
        cmd_purge(dry_run=opts["dry_run"])
    elif opts["delete"]:
        if not opts["positional"]:
            print("ERROR: --delete requires a session prefix", file=sys.stderr)
            sys.exit(1)
        cmd_delete(opts["positional"][0])
    elif opts["list"]:
        cmd_list(search=opts["search"], as_json=j)
    elif opts["smart"]:
        prefix = opts["positional"][0] if opts["positional"] else None
        cmd_smart(prefix=prefix, search=opts["search"])
    elif len(opts["positional"]) >= 2:
        cmd_rename(opts["positional"][0], " ".join(opts["positional"][1:]))
    else:
        print(__doc__)
        sys.exit(1)
