---
name: plans
description: List Claude Code plans with linked sessions
disable-model-invocation: true
allowed-tools: Bash(python3 *)
argument-hint: "[search-term]"
---

# List Plans

List all Claude Code plans with their metadata and linked sessions.

## Instructions

1. Run the script with `--plans --json` flag:

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --plans --json
```

If the user provided arguments (`$ARGUMENTS`), add `--search` with the search term:

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --plans --json --search "$ARGUMENTS"
```

2. Format the JSON result as a clean markdown table with columns:
   - Name (plan stem, truncated to 30 chars)
   - Date (modified)
   - Size (sizeKb)
   - Title (from plan file header)
   - Sessions (count + prefixes of linked sessions)

3. Show the total count at the bottom.

4. For plans with linked sessions, show the session titles as sub-items if the user asks for detail.
