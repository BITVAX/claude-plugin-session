# claude-plugin-session

Claude Code plugin for session management. List, rename, auto-title, purge, and delete sessions.

## Commands

| Command | Description |
|---------|-------------|
| `/session:list [search]` | List all sessions, optionally filtered by search term |
| `/session:name [title]` | Name the current session (auto-generates if no title given) |
| `/session:rename <prefix> <title>` | Rename any session by its prefix |
| `/session:smart` | AI-powered batch renaming of untitled sessions |
| `/session:purge` | Clean up empty sessions (dry-run first, then confirm) |
| `/session:delete <prefix>` | Delete a specific session (shows info first, then confirm) |
| `/session:themes` | Generate or update the thematic classification map |

## Hooks

- **SessionEnd**: Auto-titles sessions when closing (if untitled and >3 messages)
- **SessionStart**: Notifies about session hygiene (untitled/empty session count)

## Installation

### From local directory

```bash
/plugin marketplace add /path/to/claude-plugin-session
/plugin install session@claude-plugin-session
```

### From GitHub

```bash
/plugin marketplace add BITVAX/claude-plugin-session
/plugin install session@bitvax-claude-plugin-session
```

### Development

```bash
claude --plugin-dir /path/to/claude-plugin-session
```

## Theme Map

The theme classification is dynamic, not hardcoded. Use `/session:themes` to generate
a theme map from your existing session titles. The map is stored per-project in
`~/.claude/projects/<project>/session-themes.json`.

## CLI Usage

The script can also be used standalone:

```bash
python3 scripts/session-manager.py --list
python3 scripts/session-manager.py --status --json
python3 scripts/session-manager.py --themes
python3 scripts/session-manager.py <prefix> "New Title"
```

Project auto-detection: uses `CLAUDE_PROJECT_DIR` env var, current working directory,
or `--project-dir` flag.

## License

AGPL-3.0
