---
name: purge
description: Clean up empty sessions with 0-3 messages
disable-model-invocation: true
allowed-tools: Bash(python3 *)
---

# Purge Empty Sessions

Delete all sessions with 3 or fewer messages (just initialization, no real content).

## Instructions

### Step 1: Dry run first (ALWAYS)

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --purge --dry-run
```

### Step 2: Show the user what would be deleted

Present the results clearly: how many sessions, their sizes, message counts.

### Step 3: ASK FOR CONFIRMATION

This is a destructive operation. Ask the user to confirm before proceeding.

### Step 4: Execute (only if confirmed)

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --purge
```

### Step 5: Show results

Report how many sessions were purged and how many remain.
