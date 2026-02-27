#!/bin/bash
# SessionEnd hook: auto-title sessions that don't have a custom title.
# Runs OUTSIDE of the active Claude session, so `claude -p` works.
#
# Input (JSON via stdin):
#   session_id, transcript_path, reason (clear|logout|prompt_input_exit|other)

INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id')
REASON=$(echo "$INPUT" | jq -r '.reason // "unknown"')
TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // ""')

# Skip if session was cleared (not a real session end)
if [ "$REASON" = "clear" ]; then
  exit 0
fi

# Build args: use transcript_path to detect project dir
ARGS="--auto-title --session-id $SESSION_ID"
if [ -n "$TRANSCRIPT" ]; then
  PROJECT_DIR=$(dirname "$TRANSCRIPT")
  ARGS="$ARGS --project-dir $PROJECT_DIR"
fi

python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" $ARGS 2>/dev/null

exit 0
