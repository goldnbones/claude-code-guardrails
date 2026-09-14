# Claude Code Guardrails

`guard-destructive.py` is a free, MIT-licensed **PreToolUse hook for Claude Code** that blocks destructive shell commands before they run: `rm -rf` on `/`, `~` or wildcards, `git push --force` to protected branches, `git reset --hard`, `git clean -fd`, `DROP TABLE`, `dd of=/dev/…`, `mkfs`, `chmod -R 777`, `curl … | sh`, fork bombs and more.

It understands quoting, `&&` / `||` / `;` / `|`, subshells, `$(...)`, heredocs, `sudo -E`, `env X=1`, `nohup`, `timeout`, `xargs`, `npx`, `bash -c "..."` and `eval`. Python 3.8+ standard library only. Blocks with exit code 2 and a precise reason Claude can act on; never nags on safe commands.

## Install (free hook)

```bash
mkdir -p .claude/hooks
curl -fsSL https://goldnbones.github.io/claude-code-guardrails/free/guard-destructive.py -o .claude/hooks/guard-destructive.py
```

Add to `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Bash",
        "hooks": [ { "type": "command", "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/guard-destructive.py" } ] }
    ]
  }
}
```

Try it: ask Claude to run `rm -rf /` and watch it get refused with the rule name. Configure allow-lists, extra patterns and protected branches in `.claude/guardrails.json` (see `free/guardrails.example.json`).

## The full pack ($19)

The paid pack adds `guard-secrets.py` (stops AWS keys, private keys and API tokens being written into tracked files), a `/changelog` skill, a structured PR-review subagent, a CLAUDE.md template for Next.js + SQLite, an idempotent installer/uninstaller, and the 400-assertion test suite so you can extend rules safely.

→ https://goldnbones.github.io/claude-code-guardrails/
