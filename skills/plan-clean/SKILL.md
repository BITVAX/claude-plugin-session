---
name: plan-clean
description: Clean up orphaned plans not linked to any session
disable-model-invocation: true
allowed-tools: Bash(python3 *)
---

# Clean Orphaned Plans

Delete plans that are not linked to any session (orphans).

## Instructions

### Step 1: Dry run first (ALWAYS)

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --plan-clean --dry-run
```

### Step 2: Show the user what would be cleaned

Present the results clearly: plan names, sizes, titles.
Explain that orphaned plans are moved to `~/.claude/plans/.trash/` (not permanently deleted).

### Step 3: ASK FOR CONFIRMATION

This is a semi-destructive operation. Ask the user to confirm before proceeding.

### Step 4: Execute (only if confirmed)

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --plan-clean
```

### Step 5: Show results

Report how many plans were moved to .trash and how many remain.
