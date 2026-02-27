---
name: rename
description: Rename any session by its prefix
disable-model-invocation: true
allowed-tools: Bash(python3 *)
argument-hint: "<prefix> <new-title>"
---

# Rename Session

Rename any session by providing its prefix and new title.

## Instructions

1. Parse `$ARGUMENTS` to extract:
   - First word: session prefix (first 4-8 chars of session ID)
   - Rest: new title

2. Run the rename command:

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" <prefix> "<new-title>"
```

3. Confirm the rename was successful.

If the user doesn't provide enough arguments, show usage: `/session:rename <prefix> <new-title>`
