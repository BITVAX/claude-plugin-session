---
name: themes
description: Generate or update the session theme map from existing session titles
disable-model-invocation: true
allowed-tools: Bash(python3 *)
---

# Session Themes

Generate or update the thematic classification map for sessions.

## Instructions

YOU (Claude) generate the theme map directly — no subprocess needed.

### Step 1: Get all sessions with titles

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --list --json
```

### Step 2: Analyze the titles

Look at all session titles and identify recurring themes/topics.
Group them into categories (max 15 themes).

For each theme, identify keywords that appear in the titles:
- Use lowercase keywords
- Include both languages if sessions mix Spanish/English
- Be specific enough to avoid false positives

### Step 3: Build the theme map

Create a JSON structure:

```json
{
  "themes": [
    {"name": "ThemeName", "keywords": ["keyword1", "keyword2", "keyword3"]},
    ...
  ],
  "lastUpdated": "YYYY-MM-DD"
}
```

### Step 4: Save it

Write the JSON to the session-themes.json file in the project sessions directory.
The path is shown by:

```
python3 -c "
import os
from pathlib import Path
cwd = os.getcwd()
slug = cwd.replace('/', '-')
print(Path.home() / '.claude/projects' / slug / 'session-themes.json')
"
```

### Step 5: Verify

Run list again to verify themes appear in the output:

```
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/session-manager.py" --list
```

Show the user the theme map and the themed session list.
