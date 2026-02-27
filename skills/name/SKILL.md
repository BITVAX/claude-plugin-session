---
name: name
description: Set a custom title for the current session
disable-model-invocation: true
allowed-tools: Bash(python3 *)
argument-hint: "[title]"
---

# Name Current Session

Set a custom title for the currently active session.

## Instructions

1. Get the current session ID. It is available as `${CLAUDE_SESSION_ID}` in the environment.

2. If the user provided a title in `$ARGUMENTS`, apply it directly:

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --project-dir "$(dirname "${CLAUDE_TRANSCRIPT_PATH:-/dev/null}")" "${CLAUDE_SESSION_ID}" "$ARGUMENTS"
```

3. If NO title was provided, analyze the current conversation and generate a short descriptive title (max 60 chars, in the same language the user has been using). Then apply it with the command above.

4. Confirm the title was set.
