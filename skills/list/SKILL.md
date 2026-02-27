---
name: list
description: List Claude Code sessions with optional search filter
disable-model-invocation: true
allowed-tools: Bash(python3 *)
argument-hint: "[search-term]"
---

# List Sessions

Run the session manager to list all sessions for the current project.

## Instructions

1. Run the script with `--list --json` flag:

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --list --json
```

If the user provided arguments (`$ARGUMENTS`), add `--search` with the search term:

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --list --json --search "$ARGUMENTS"
```

2. Format the JSON result as a clean markdown table with columns:
   - Prefix (first 8 chars of sessionId)
   - Date (created)
   - Msgs (messageCount)
   - Title (summary or firstPrompt, mark with * if hasCustomTitle)
   - Theme (if available)

3. Show the total count at the bottom.
