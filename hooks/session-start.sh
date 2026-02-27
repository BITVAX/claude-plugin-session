#!/bin/bash
# SessionStart hook: notify about session hygiene (untitled/empty sessions).
#
# Input (JSON via stdin):
#   session_id, transcript_path, source (startup|resume|clear|compact)
#
# Only runs on fresh startup (not resume/clear/compact).

INPUT=$(cat)
SOURCE=$(echo "$INPUT" | jq -r '.source // "unknown"')

# Only on fresh startup
if [ "$SOURCE" != "startup" ]; then
  exit 0
fi

# Detect project dir from transcript_path
TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // ""')
EXTRA_ARGS=""
if [ -n "$TRANSCRIPT" ]; then
  PROJECT_DIR=$(dirname "$TRANSCRIPT")
  EXTRA_ARGS="--project-dir $PROJECT_DIR"
fi

STATUS=$(python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --status --json $EXTRA_ARGS 2>/dev/null)

if [ -z "$STATUS" ]; then
  exit 0
fi

UNTITLED=$(echo "$STATUS" | jq -r '.untitled // 0')
EMPTY=$(echo "$STATUS" | jq -r '.empty // 0')

MSGS=""
[ "$UNTITLED" -gt 5 ] && MSGS="${UNTITLED} untitled sessions"
[ "$EMPTY" -gt 3 ] && { [ -n "$MSGS" ] && MSGS="$MSGS, "; MSGS="${MSGS}${EMPTY} empty (run /session:purge)"; }

if [ -n "$MSGS" ]; then
  echo "{\"hookSpecificOutput\":{\"hookEventName\":\"SessionStart\",\"additionalContext\":\"Session hygiene: $MSGS\"}}"
fi

exit 0
