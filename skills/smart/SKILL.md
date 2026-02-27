---
name: smart
description: AI-powered batch renaming of untitled sessions
disable-model-invocation: true
allowed-tools: Bash(python3 *)
context: fork
---

# Smart Rename Sessions

Automatically generate titles for sessions that don't have one yet.

## Instructions

YOU (Claude) generate the titles directly — no need for `claude -p` subprocess.

### Step 1: Get untitled sessions

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --list-untitled --json
```

### Step 2: For each untitled session, read its messages

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --messages <sessionId> --json
```

### Step 3: Generate a title

Based on the messages, generate a short descriptive title:
- Max 60 characters
- In the language the user was using in that session
- Summarize the main topic/task of the session
- Be specific (not generic like "Development session")

### Step 4: Apply the title

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" <prefix> "<generated-title>"
```

### Step 5: Report results

Show a summary of all renamed sessions as a table.

IMPORTANT: Process sessions one by one. If there are many (>20), ask the user before proceeding.
