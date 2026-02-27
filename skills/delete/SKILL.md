---
name: delete
description: Delete a specific session by its prefix
disable-model-invocation: true
allowed-tools: Bash(python3 *)
argument-hint: "<session-prefix>"
---

# Delete Session

Delete a specific session identified by its prefix.

## Instructions

### Step 1: Show session info first

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --info $ARGUMENTS --json
```

### Step 2: Present the info to the user

Show: session ID, title, date, message count, size.

### Step 3: ASK FOR CONFIRMATION

This is a destructive operation. Ask the user explicitly: "Delete this session?"

### Step 4: Execute (only if confirmed)

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --delete $ARGUMENTS
```

### Step 5: Confirm deletion
