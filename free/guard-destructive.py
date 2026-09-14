#!/usr/bin/env python3
"""
guard-destructive.py -- Claude Code PreToolUse hook for the Bash tool.

Blocks destructive shell commands before Claude Code executes them.

Contract (Claude Code hooks):
    stdin   JSON: {"session_id": "...", "tool_name": "Bash",
                   "tool_input": {"command": "..."}, "cwd": "..."}
    exit 0  allow the tool call
    exit 2  BLOCK the tool call; stderr is shown to Claude as the reason

What it catches (see README "What it blocks" for the full table):
    * rm -r/-rf on /, ~, system dirs, bare wildcards, `$VAR/` targets, .git
    * sudo rm (any form), shred/unlink of protected files
    * truncation of protected files (> .env, truncate, tee, cp /dev/null ...)
    * git push --force / +refspec / --mirror / --delete on protected branches
    * git reset --hard, git clean -f, git checkout -- ., git restore .,
      git stash clear/drop, filter-branch, reflog expire, gc --prune=now
    * SQL DROP TABLE/DATABASE/SCHEMA, TRUNCATE, DELETE FROM x (no WHERE)
      when handed to a database client (psql, mysql, sqlite3, prisma ...)
    * database / infra destroy commands (dropdb, prisma migrate reset,
      redis-cli flushall, terraform destroy, aws s3 rb, kubectl delete ns ...)
    * chmod -R 777 / world-writable, chmod/chown -R on protected paths
    * dd of=/dev/..., > /dev/sdX, mkfs, fdisk, wipefs, diskutil erase
    * fork bombs, curl | sh style remote-code piping
    * shutdown/reboot, kill -9 -1, crontab -r
    * user-defined block_patterns

Robustness: the command is tokenised with shell-quoting awareness, split on
&& || ; | & and newlines, subshells and $(...)/`...` substitutions are
analysed recursively, wrapper prefixes (sudo -E, env X=1, nohup, timeout 5,
xargs, npx, command, exec, bash -c "...", eval ...) are stripped so the real
command is examined. Malformed stdin never crashes the hook (fail-open by
default, configurable).

Configuration: .claude/guardrails.json (project) and ~/.claude/guardrails.json
(user), section "destructive". See guardrails.example.json.

Dependency-free: Python 3.8+ standard library only.
"""

import fnmatch
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys

HOOK_NAME = "guard-destructive"
MAX_RECURSION_DEPTH = 6

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG = {
    # Branch names / fnmatch patterns that must never be force-pushed or deleted.
    "protected_branches": [
        "main", "master", "develop", "production", "prod", "staging",
        "release", "release/*",
    ],
    # Extra paths (absolute, ~-relative, or relative) that rm -r may not touch.
    "protected_paths": [],
    # Extra glob patterns for files that must not be truncated/deleted.
    "protected_files": [],
    # Regexes (re.search) -- a command matching any of these is ALWAYS allowed.
    "allow_patterns": [],
    # Regexes (re.search) -- a command matching any of these is ALWAYS blocked.
    "block_patterns": [],
    # Rule ids to switch off, e.g. ["git_reset_hard", "db_file_delete"].
    "disabled_rules": [],
    # Treat --force-with-lease to a protected branch as OK.
    "allow_force_with_lease": False,
    # Also block *non-force* pushes straight to protected branches.
    "block_direct_push_to_protected": False,
    # If stdin is unreadable/malformed, block (True) or allow (False).
    "fail_closed": False,
}

LIST_KEYS = {"protected_branches", "protected_paths", "protected_files",
             "allow_patterns", "block_patterns", "disabled_rules"}


def _read_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_config(cwd):
    """Merge defaults <- ~/.claude/guardrails.json <- project guardrails.json.

    Accepts either {"destructive": {...}} or a flat object. List keys are
    concatenated (deduplicated); scalar keys are overridden.
    """
    cfg = {k: (list(v) if isinstance(v, list) else v)
           for k, v in DEFAULT_CONFIG.items()}
    candidates = []
    home = os.path.expanduser("~")
    candidates.append(os.path.join(home, ".claude", "guardrails.json"))
    proj = os.environ.get("CLAUDE_PROJECT_DIR")
    if proj:
        candidates.append(os.path.join(proj, ".claude", "guardrails.json"))
    if cwd:
        candidates.append(os.path.join(cwd, ".claude", "guardrails.json"))
    env_path = os.environ.get("GUARDRAILS_CONFIG")
    if env_path:
        candidates.append(env_path)

    seen = set()
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        data = _read_json_file(path)
        section = data.get("destructive", data)
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if key.startswith("_"):
                continue
            if key in LIST_KEYS and isinstance(value, list):
                for item in value:
                    # Strings starting with "_" are documentation, not config.
                    if isinstance(item, str) and not item.startswith("_") and item not in cfg[key]:
                        cfg[key].append(item)
            elif key in DEFAULT_CONFIG and not isinstance(DEFAULT_CONFIG[key], list):
                cfg[key] = value
    return cfg


# --------------------------------------------------------------------------- #
# Shell parsing helpers
# --------------------------------------------------------------------------- #

def preprocess(cmd):
    """Join line continuations and pull heredoc bodies out of the command.

    Returns (command_without_heredoc_bodies, [(owner_line, body), ...]).
    """
    cmd = cmd.replace("\r\n", "\n").replace("\\\n", " ")
    lines = cmd.split("\n")
    out_lines, heredocs = [], []
    i = 0
    heredoc_re = re.compile(r"<<-?\s*(?:\"([^\"]+)\"|'([^']+)'|\\?([A-Za-z_][A-Za-z0-9_]*))")
    while i < len(lines):
        line = lines[i]
        m = heredoc_re.search(line)
        if m:
            term = m.group(1) or m.group(2) or m.group(3)
            body = []
            i += 1
            while i < len(lines) and lines[i].strip() != term:
                body.append(lines[i])
                i += 1
            heredocs.append((line, "\n".join(body)))
            out_lines.append(line)
            i += 1
            continue
        out_lines.append(line)
        i += 1
    return "\n".join(out_lines), heredocs


def extract_substitutions(cmd):
    """Return inner strings of $( ... ), <( ... ), >( ... ) and `...`."""
    found = []
    n = len(cmd)
    i = 0
    in_single = False
    while i < n:
        ch = cmd[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "'" and not in_single:
            in_single = True
            i += 1
            continue
        if ch == "'" and in_single:
            in_single = False
            i += 1
            continue
        if in_single:
            i += 1
            continue
        if ch in "$<>" and i + 1 < n and cmd[i + 1] == "(":
            depth, j = 0, i + 1
            while j < n:
                if cmd[j] == "\\":
                    j += 2
                    continue
                if cmd[j] == "(":
                    depth += 1
                elif cmd[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            inner = cmd[i + 2:j]
            found.append(inner)
            i = j + 1
            continue
        if ch == "`":
            j = cmd.find("`", i + 1)
            if j == -1:
                break
            found.append(cmd[i + 1:j])
            i = j + 1
            continue
        i += 1
    return found


def split_segments(cmd):
    """Split on unquoted control operators.

    Returns a list of (connector, segment_text). connector is the operator
    that preceded the segment: '' | '&&' | '||' | ';' | '|' | '&' | '\\n'
    | '(' | ')'.
    """
    segments = []
    buf = []
    connector = ""
    in_single = in_double = False
    i, n = 0, len(cmd)

    def flush(next_connector):
        nonlocal buf, connector
        text = "".join(buf).strip()
        if text:
            segments.append((connector, text))
        buf = []
        connector = next_connector

    while i < n:
        ch = cmd[i]
        nxt = cmd[i + 1] if i + 1 < n else ""
        if ch == "\\" and not in_single:
            buf.append(ch)
            if i + 1 < n:
                buf.append(nxt)
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            i += 1
            continue
        if in_single or in_double:
            buf.append(ch)
            i += 1
            continue
        # Comments: '#' at start of a word ends the segment's meaningful text.
        if ch == "#" and (i == 0 or cmd[i - 1] in " \t\n;|&("):
            j = cmd.find("\n", i)
            i = n if j == -1 else j
            continue
        if ch == "\n":
            flush("\n")
            i += 1
            continue
        if ch == "&":
            prev = cmd[i - 1] if i > 0 else ""
            if nxt == "&":
                flush("&&")
                i += 2
                continue
            if (prev and prev in "<>") or nxt == ">":  # 2>&1, &>file, &>>file
                buf.append(ch)
                i += 1
                continue
            flush("&")
            i += 1
            continue
        if ch == "|":
            prev = cmd[i - 1] if i > 0 else ""
            if prev == ">":                # `>|` forces clobber; not a pipe
                buf.append(ch)
                i += 1
                continue
            if nxt == "|":
                flush("||")
                i += 2
                continue
            if nxt == "&":
                flush("|")
                i += 2
                continue
            flush("|")
            i += 1
            continue
        if ch == ";":
            flush(";")
            i += 2 if nxt == ";" else 1
            continue
        if ch == "(":
            prev = cmd[i - 1] if i > 0 else ""
            if prev and prev in "$<>":
                buf.append(ch)
                i += 1
                continue
            flush("(")
            i += 1
            continue
        if ch == ")":
            flush(")")
            i += 1
            continue
        buf.append(ch)
        i += 1
    flush("")
    return segments


def tokenize(segment):
    """shlex.split with graceful fallback for unbalanced quotes."""
    for candidate in (segment, segment + "'", segment + '"', segment + "'\""):
        try:
            return shlex.split(candidate, posix=True)
        except ValueError:
            continue
    return segment.replace('"', " ").replace("'", " ").split()


REDIRECT_RE = re.compile(r"^(\d*|&)?(>>|>\||>|<<<|<<|<)(.*)$")


def strip_redirections(tokens):
    """Remove redirection operators; return (clean_tokens, [(op, target)])."""
    clean, redirects = [], []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        m = REDIRECT_RE.match(tok)
        if m and not tok.startswith("<(") and not tok.startswith(">("):
            op, target = m.group(2), m.group(3)
            if target == "" and i + 1 < len(tokens):
                target = tokens[i + 1]
                i += 1
            if target.startswith("&"):  # 2>&1 -- fd duplication
                i += 1
                continue
            redirects.append((op, target))
            i += 1
            continue
        clean.append(tok)
        i += 1
    return clean, redirects


# Wrapper commands that prefix the "real" command. Value: (short opts that
# take an argument, long opts that take an argument, number of positional
# arguments to skip).
WRAPPERS = {
    "sudo": ("ugpCrthU", {"--user", "--group", "--prompt", "--chdir",
                          "--role", "--type", "--host", "--other-user",
                          "--close-from"}, 0),
    "doas": ("uC", set(), 0),
    "su": ("", set(), 0),               # handled specially (-c)
    "env": ("uCS", {"--unset", "--chdir", "--split-string"}, 0),
    "nohup": ("", set(), 0),
    "nice": ("n", {"--adjustment"}, 0),
    "ionice": ("cnp", {"--class", "--classdata", "--pid"}, 0),
    "time": ("f", {"--format", "--output"}, 0),
    "timeout": ("sk", {"--signal", "--kill-after"}, 1),
    "command": ("", set(), 0),
    "exec": ("a", set(), 0),
    "builtin": ("", set(), 0),
    "xargs": ("InPdLsEai", {"--max-args", "--max-procs", "--delimiter",
                            "--max-lines", "--max-chars", "--eof",
                            "--arg-file", "--replace"}, 0),
    "busybox": ("", set(), 0),
    "caffeinate": ("tw", set(), 0),
    "stdbuf": ("oei", {"--output", "--error", "--input"}, 0),
    "unbuffer": ("", set(), 0),
    "chronic": ("", set(), 0),
    "setsid": ("", set(), 0),
    "flock": ("wE", {"--timeout", "--conflict-exit-code"}, 1),
    "chroot": ("ug", {"--userspec", "--groups"}, 1),
    "watch": ("nd", {"--interval", "--differences"}, 0),
    "npx": ("pc", {"--package", "--call"}, 0),
    "bunx": ("p", set(), 0),
    "strace": ("oepf", {"--output"}, 0),
    "ltrace": ("oe", set(), 0),
    "script": ("c", set(), 0),          # handled specially (-c)
    "dotenv": ("ef", {"--env-file"}, 0),
}
# Two-word wrappers: first word -> set of second words that make it a wrapper.
TWO_WORD_WRAPPERS = {
    "pnpm": {"dlx", "exec"},
    "yarn": {"dlx", "exec"},
    "npm": {"exec"},
    "poetry": {"run"},
    "pipenv": {"run"},
    "uv": {"run"},
    "pdm": {"run"},
    "rye": {"run"},
    "bundle": {"exec"},
    "bundler": {"exec"},
    "mise": {"exec", "x"},
    "asdf": {"exec"},
    "dotenv": {"run"},
    "hatch": {"run"},
}
SHELL_KEYWORDS = {"{", "}", "!", "if", "then", "else", "elif", "fi", "do",
                  "done", "while", "until", "case", "esac", "in", "for",
                  "select", "function", "coproc", "--"}
ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\+?=")


def base_name(token):
    """`/usr/bin/rm` -> `rm`, `\\rm` -> `rm`, `python3.11` stays as-is."""
    token = token.lstrip("\\")
    return posixpath.basename(token.rstrip("/")) or token


def strip_wrappers(tokens):
    """Peel wrappers/keywords/assignments off the front of a token list.

    Returns (base_command, args, meta) where meta has: sudo (bool),
    xargs (bool), wrappers (list of names seen).
    """
    meta = {"sudo": False, "xargs": False, "wrappers": []}
    toks = list(tokens)
    guard = 0
    while toks and guard < 50:
        guard += 1
        head = toks[0]
        if head in SHELL_KEYWORDS or ASSIGNMENT_RE.match(head):
            toks.pop(0)
            continue
        name = base_name(head)
        if name in TWO_WORD_WRAPPERS and len(toks) > 1 and toks[1] in TWO_WORD_WRAPPERS[name]:
            meta["wrappers"].append(name + " " + toks[1])
            toks = toks[2:]
            while toks and toks[0].startswith("-") and toks[0] != "--":
                toks.pop(0)
            if toks and toks[0] == "--":
                toks.pop(0)
            continue
        if name not in WRAPPERS:
            break
        if name == "command" and len(toks) > 1 and toks[1] in ("-v", "-V"):
            return "", [], meta            # `command -v foo` is a lookup, not exec
        if name in ("su", "script"):
            break                          # analysed by nested-command logic
        short_arg, long_arg, positional = WRAPPERS[name]
        meta["wrappers"].append(name)
        if name in ("sudo", "doas"):
            meta["sudo"] = True
        if name == "xargs":
            meta["xargs"] = True
        toks.pop(0)
        while toks:
            t = toks[0]
            if t == "--":
                toks.pop(0)
                break
            if t.startswith("--"):
                toks.pop(0)
                if "=" not in t and t in long_arg and toks:
                    toks.pop(0)
                continue
            if t.startswith("-") and len(t) > 1:
                toks.pop(0)
                # short cluster: last letter may take an argument (-Eu root)
                if t[-1] in short_arg and toks and len(t) == 2:
                    toks.pop(0)
                elif t[-1] in short_arg and toks and len(t) > 2 and all(c not in short_arg for c in t[1:-1]):
                    toks.pop(0)
                continue
            if name == "sudo" and ASSIGNMENT_RE.match(t):
                toks.pop(0)                # sudo VAR=value cmd
                continue
            break
        for _ in range(positional):
            if toks:
                toks.pop(0)
        continue
    if not toks:
        return "", [], meta
    return base_name(toks[0]), toks[1:], meta


# --------------------------------------------------------------------------- #
# Path classification
# --------------------------------------------------------------------------- #

PROTECTED_TREES = ["/etc", "/boot", "/bin", "/sbin", "/lib", "/lib32",
                   "/lib64", "/dev", "/proc", "/sys", "/System", "/cores",
                   "/nix", "/snap", "/run", "/private/etc", "/private/var"]
PROTECTED_DIRS = ["/", "/usr", "/var", "/opt", "/home", "/Users", "/root",
                  "/Library", "/Applications", "/Volumes", "/private", "/srv",
                  "/mnt", "/media"]
# Protected as a whole, but children (e.g. /tmp/build) are fair game.
PROTECTED_EXACT = ["/tmp", "/var/tmp", "/private/tmp"]
PROTECTED_HOME = ["~", "~/.ssh", "~/.gnupg", "~/.aws", "~/.config", "~/.claude",
                  "~/.kube", "~/.docker", "~/Desktop", "~/Documents",
                  "~/Downloads", "~/Pictures", "~/Movies", "~/Music",
                  "~/Library", "~/Applications"]
PROTECTED_RELATIVE = [".", "..", ".git"]

PROTECTED_FILE_GLOBS = [
    ".env", ".env.*", "*/.env", "*/.env.*",
    "~/.ssh/*", "~/.gnupg/*", "~/.aws/*", "~/.bashrc", "~/.zshrc",
    "~/.bash_profile", "~/.profile", "~/.zprofile", "~/.gitconfig",
    "~/.netrc", "~/.npmrc", "~/.pypirc",
    "/etc/*", "/etc/*/*", "/boot/*",
    "*.db", "*.sqlite", "*.sqlite3", "*.db-wal", "*.db-shm",
    ".git/HEAD", ".git/config", ".git/packed-refs", ".git/objects", ".git/refs",
    ".git/objects/*", ".git/refs/*", "*/.git/HEAD", "*/.git/config", "*/.git/objects", "*/.git/refs",
    "*.pem", "*.key", "*.p12", "*.pfx", "*.keystore",
]
EXAMPLE_FILE_SUFFIXES = (".example", ".sample", ".template", ".dist", ".md")


def _home_dir():
    return os.path.expanduser("~").rstrip("/")


def normalize_path(token, cwd=None):
    """Normalise a path token for comparison.

    Home is reduced to '~', multiple slashes collapsed, trailing slashes and
    ./ components removed. Returns (normalized, had_trailing_slash).
    """
    raw = token
    had_trailing_slash = len(raw) > 1 and raw.endswith("/")
    home = _home_dir()
    for var in ("${HOME}", "$HOME"):
        if raw.startswith(var):
            raw = "~" + raw[len(var):]
    if raw.startswith("$PWD") or raw.startswith("${PWD}"):
        raw = "." + raw.split("}", 1)[-1] if raw.startswith("${") else "." + raw[4:]
    if raw.startswith("~/") or raw == "~":
        rest = raw[1:]
        raw = "~" + (posixpath.normpath(rest) if rest not in ("", "/") else "")
        raw = raw.rstrip("/") if raw != "~" else raw
        raw = "~" if raw in ("~/.", "~") else raw
    else:
        collapsed = re.sub(r"/+", "/", raw)
        norm = posixpath.normpath(collapsed) if collapsed else collapsed
        if norm.startswith("//"):
            norm = norm[1:]
        if home and (norm == home or norm.startswith(home + "/")):
            norm = "~" + norm[len(home):]
        raw = norm
    return raw, had_trailing_slash


def resolve_against_cwd(norm, cwd):
    """Best-effort absolute (home-reduced) form of a relative path."""
    if not cwd or norm.startswith(("/", "~")):
        return None
    joined = posixpath.normpath(posixpath.join(cwd, norm))
    resolved, _ = normalize_path(joined)
    return resolved


def classify_protected_dir(token, cfg, cwd=None):
    """Return a human-readable reason if `token` names a protected directory
    (for recursive operations), else None."""
    if token in ("", "-", "--"):
        return None
    norm, _ = normalize_path(token, cwd)
    candidates = [norm]
    resolved = resolve_against_cwd(norm, cwd)
    if resolved and resolved != norm:
        candidates.append(resolved)

    for cand in candidates:
        if cand == "/":
            return "the filesystem root (/)"
        if cand == "~":
            return "the home directory (~)"
        if cand in PROTECTED_RELATIVE and cand == norm:
            label = {".": "the current directory (.)", "..": "the parent directory (..)",
                     ".git": "the .git repository directory"}[cand]
            return label
        if cand.endswith("/.git") and cand == norm or cand == ".git":
            return "a .git repository directory"
        for tree in PROTECTED_TREES:
            if cand == tree or cand.startswith(tree + "/"):
                return "the protected system tree %s" % tree
        if cand in PROTECTED_EXACT:
            return "the shared temp directory %s" % cand
        for d in PROTECTED_DIRS:
            if cand == d:
                return "the protected system directory %s" % d
            # depth-2 children of system dirs, e.g. /usr/local, /Users/alice
            if d != "/" and cand.startswith(d + "/") and cand.count("/") == d.count("/") + 1:
                return "the protected system directory %s" % cand
        for h in PROTECTED_HOME:
            if cand == h:
                return "the protected home directory %s" % h
        for extra in cfg.get("protected_paths", []):
            e_norm, _ = normalize_path(extra, cwd)
            if cand == e_norm or fnmatch.fnmatch(cand, e_norm):
                return "the configured protected path %s" % extra
    return None


def is_bare_wildcard(token):
    """`*`, `.*`, `**`, `./*`, `/*`, `~/*`, `$HOME/*`, `src/../*` ..."""
    base = posixpath.basename(token.rstrip("/")) if token not in ("*", ".*", "**") else token
    return base in ("*", ".*", "**", "*.*")


def wildcard_dir_is_protected(token, cfg, cwd):
    """For `<dir>/*` return a reason if <dir> is bare cwd or protected."""
    if not is_bare_wildcard(token):
        return None
    dirname = posixpath.dirname(token.rstrip("/")) if "/" in token else ""
    if dirname in ("", "."):
        return "everything in the current directory (bare wildcard %s)" % token
    reason = classify_protected_dir(dirname, cfg, cwd)
    if reason:
        return "everything inside %s (wildcard %s)" % (reason, token)
    return None


VAR_PREFIX_RE = re.compile(r"^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?/(\*|\.\*)?$")


def is_protected_file(token, cfg, cwd=None):
    """True if token matches a protected-file glob (truncation/deletion)."""
    if not token or token.startswith("-"):
        return False
    norm, _ = normalize_path(token, cwd)
    if norm.endswith(EXAMPLE_FILE_SUFFIXES):
        return False
    candidates = [norm]
    resolved = resolve_against_cwd(norm, cwd)
    if resolved:
        candidates.append(resolved)
    base = posixpath.basename(norm)
    globs = PROTECTED_FILE_GLOBS + list(cfg.get("protected_files", []))
    for g in globs:
        g_norm, _ = normalize_path(g) if not g.startswith("~") else (g, False)
        for cand in candidates:
            if fnmatch.fnmatchcase(cand, g_norm) or fnmatch.fnmatchcase(cand, g):
                return True
        if "/" not in g and fnmatch.fnmatchcase(base, g):
            return True
    return False


# --------------------------------------------------------------------------- #
# Rule helpers
# --------------------------------------------------------------------------- #

class Block(Exception):
    """Raised by a rule to block the command."""

    def __init__(self, rule, reason):
        super().__init__(reason)
        self.rule = rule
        self.reason = reason


def rule_enabled(cfg, rule_id):
    return rule_id not in cfg.get("disabled_rules", [])


def block(cfg, rule_id, reason):
    if rule_enabled(cfg, rule_id):
        raise Block(rule_id, reason)


def short_flags(args, stop_at_double_dash=True):
    """Collect letters from short-option clusters (-rf -> {'r','f'})."""
    letters = set()
    for a in args:
        if a == "--" and stop_at_double_dash:
            break
        if a.startswith("-") and not a.startswith("--") and len(a) > 1:
            letters.update(a[1:])
    return letters


def positional_args(args):
    """Arguments not starting with '-' (everything after '--' is positional)."""
    out = []
    after_dd = False
    for a in args:
        if after_dd:
            out.append(a)
        elif a == "--":
            after_dd = True
        elif not a.startswith("-") or a == "-":
            out.append(a)
    return out


def branch_protected(name, cfg):
    if not name:
        return False
    name = re.sub(r"^refs/heads/", "", name)
    return any(fnmatch.fnmatchcase(name, pat) for pat in cfg.get("protected_branches", []))


def current_branch(cwd, git_dir_args):
    """Resolve the checked-out branch, or None."""
    try:
        cmd = ["git"] + git_dir_args + ["rev-parse", "--abbrev-ref", "HEAD"]
        res = subprocess.run(cmd, cwd=cwd or None, capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            name = res.stdout.strip()
            return name if name and name != "HEAD" else None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


# --------------------------------------------------------------------------- #
# Rules: rm / deletion / truncation
# --------------------------------------------------------------------------- #

def check_rm(base, args, meta, cfg, cwd):
    if base not in ("rm", "rmdir", "unlink", "shred", "srm", "wipe"):
        return
    flags = short_flags(args)
    longs = {a for a in args if a.startswith("--")}
    recursive = base in ("rm", "srm", "wipe") and (
        "r" in flags or "R" in flags or "--recursive" in longs)
    if base == "rm" and "--no-preserve-root" in longs:
        block(cfg, "rm_no_preserve_root",
              "`rm --no-preserve-root` is only ever used to destroy the root filesystem")
    if base == "rm" and meta["sudo"]:
        block(cfg, "sudo_rm",
              "`sudo rm` deletes files with root privileges; run deletions unprivileged and scoped to the project")
    targets = positional_args(args)
    for t in targets:
        # Bare wildcards are blocked whether or not -r is present.
        wild = wildcard_dir_is_protected(t, cfg, cwd)
        if wild:
            block(cfg, "rm_wildcard", "`%s` would delete %s" % (base, wild))
        if recursive:
            reason = classify_protected_dir(t, cfg, cwd)
            if reason:
                block(cfg, "rm_recursive_protected",
                      "recursive `%s` targeting %s" % (base, reason))
            if VAR_PREFIX_RE.match(t):
                block(cfg, "rm_variable_prefix",
                      "`%s -r %s`: if the variable is empty this expands to the filesystem root" % (base, t))
        if is_protected_file(t, cfg, cwd):
            rule = "db_file_delete" if re.search(r"\.(db|sqlite3?|db-wal|db-shm)$", t) else "protected_file_delete"
            block(cfg, rule, "`%s` would delete the protected file %s" % (base, t))


def check_truncation(base, args, redirects, meta, cfg, cwd):
    # Redirections: `> file`, `>| file` truncate; `>>` appends (allowed).
    for op, target in redirects:
        if op in (">", ">|"):
            if target.startswith("/dev/") and target not in ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty", "/dev/fd/1", "/dev/fd/2"):
                block(cfg, "write_block_device",
                      "redirecting output into device %s would overwrite raw disk contents" % target)
            if is_protected_file(target, cfg, cwd):
                block(cfg, "protected_file_truncate",
                      "`> %s` truncates/overwrites a protected file (use >> to append, or write via the Edit tool)" % target)
    if base == "truncate":
        for t in positional_args(args):
            if is_protected_file(t, cfg, cwd):
                block(cfg, "protected_file_truncate", "`truncate` would empty the protected file %s" % t)
    if base == "tee":
        if "a" not in short_flags(args) and "--append" not in args:
            for t in positional_args(args):
                if is_protected_file(t, cfg, cwd):
                    block(cfg, "protected_file_truncate", "`tee %s` overwrites a protected file (use tee -a to append)" % t)
    if base in ("cp", "install") and args:
        pos = positional_args(args)
        if len(pos) >= 2 and pos[0] in ("/dev/null", "/dev/zero") and is_protected_file(pos[-1], cfg, cwd):
            block(cfg, "protected_file_truncate", "`cp %s %s` empties a protected file" % (pos[0], pos[-1]))
    if base == "mv":
        for t in positional_args(args)[:-1]:
            reason = classify_protected_dir(t, cfg, cwd)
            if reason:
                block(cfg, "mv_protected", "`mv` would relocate %s" % reason)
    if base == "dd":
        for a in args:
            if a.startswith("of="):
                target = a[3:]
                if target.startswith("/dev/") and target not in ("/dev/null", "/dev/stdout", "/dev/stderr"):
                    block(cfg, "dd_device", "`dd of=%s` writes raw bytes to a device" % target)
                if is_protected_file(target, cfg, cwd):
                    block(cfg, "protected_file_truncate", "`dd of=%s` overwrites a protected file" % target)


def check_find(base, args, meta, cfg, cwd):
    if base != "find":
        return
    destructive = "-delete" in args or any(
        a in ("-exec", "-execdir", "-ok", "-okdir") and i + 1 < len(args)
        and base_name(args[i + 1]) in ("rm", "shred", "unlink")
        for i, a in enumerate(args))
    if not destructive:
        return
    start_paths = []
    for a in args:
        if a.startswith("-") and a not in ("-",):
            break
        start_paths.append(a)
    if not start_paths:
        start_paths = ["."]
    for p in start_paths:
        reason = classify_protected_dir(p, cfg, cwd)
        # `find . -name '*.pyc' -delete` is fine; only block protected roots
        # other than the bare current directory.
        if reason and p not in (".", "./"):
            block(cfg, "find_delete_protected", "`find %s ... -delete/-exec rm` walks %s" % (p, reason))
        if p in (".", "./") and not any(a in ("-name", "-iname", "-path", "-ipath", "-regex", "-iregex", "-type", "-newer", "-mtime", "-mmin", "-empty", "-size") for a in args):
            block(cfg, "find_delete_unfiltered", "`find . -delete` with no filter would delete the whole working tree")


# --------------------------------------------------------------------------- #
# Rules: git
# --------------------------------------------------------------------------- #

GIT_GLOBAL_OPTS_WITH_ARG = {"-C", "-c", "--git-dir", "--work-tree", "--namespace",
                            "--exec-path", "--super-prefix", "--config-env"}


def parse_git(args):
    """Return (subcommand, sub_args, global_dir_args)."""
    i = 0
    dir_args = []
    while i < len(args):
        a = args[i]
        if not a.startswith("-"):
            return a, args[i + 1:], dir_args
        if a in GIT_GLOBAL_OPTS_WITH_ARG and i + 1 < len(args):
            if a in ("-C", "--git-dir", "--work-tree"):
                dir_args += [a, args[i + 1]]
            i += 2
            continue
        i += 1
    return "", [], dir_args


def check_git(base, args, meta, cfg, cwd):
    if base != "git":
        return
    sub, rest, dir_args = parse_git(args)
    if not sub:
        return
    flags = short_flags(rest)
    longs = {a.split("=", 1)[0] for a in rest if a.startswith("--")}

    if sub == "push":
        _check_git_push(rest, flags, longs, dir_args, cfg, cwd)
    elif sub == "reset":
        if "--hard" in longs:
            block(cfg, "git_reset_hard",
                  "`git reset --hard` discards uncommitted work irreversibly (use `git stash` or `git reset --soft`)")
    elif sub == "clean":
        force = "f" in flags or "--force" in longs
        dry = "n" in flags or "--dry-run" in longs
        interactive = "i" in flags or "--interactive" in longs
        if force and not dry and not interactive:
            block(cfg, "git_clean_force",
                  "`git clean -f` permanently deletes untracked files (run `git clean -n` first and ask the user)")
    elif sub == "checkout":
        pos = positional_args(rest)
        if "f" in flags or "--force" in longs:
            block(cfg, "git_discard_worktree", "`git checkout --force` throws away local modifications")
        if any(p in (".", ":/", "*", "./") for p in pos):
            block(cfg, "git_discard_worktree",
                  "`git checkout -- .` discards every uncommitted change in the working tree")
    elif sub == "restore":
        pos = positional_args(rest)
        staged_only = ("--staged" in longs or "S" in flags) and not ("--worktree" in longs or "W" in flags)
        if any(p in (".", ":/", "*", "./") for p in pos) and not staged_only:
            block(cfg, "git_discard_worktree",
                  "`git restore .` discards every uncommitted change in the working tree")
    elif sub == "stash":
        if rest and rest[0] in ("clear", "drop"):
            block(cfg, "git_stash_destroy", "`git stash %s` permanently deletes stashed work" % rest[0])
    elif sub == "branch":
        deleting = "D" in flags or "d" in flags or "--delete" in longs
        for p in positional_args(rest):
            if deleting and branch_protected(p, cfg):
                block(cfg, "git_delete_protected_branch", "deleting protected branch `%s`" % p)
    elif sub in ("filter-branch", "filter-repo"):
        block(cfg, "git_history_rewrite", "`git %s` rewrites repository history" % sub)
    elif sub == "reflog":
        if rest and rest[0] == "expire" and any(a.startswith(("--expire=now", "--expire=all", "--expire-unreachable=now", "--expire-unreachable=all")) for a in rest):
            block(cfg, "git_history_rewrite", "`git reflog expire --expire=now` makes lost commits unrecoverable")
    elif sub == "gc":
        if any(a in ("--prune=now", "--prune=all") for a in rest):
            block(cfg, "git_history_rewrite", "`git gc --prune=now` permanently removes unreachable commits")
    elif sub == "update-ref":
        if "-d" in rest:
            for p in positional_args(rest):
                if branch_protected(p, cfg):
                    block(cfg, "git_delete_protected_branch", "`git update-ref -d` on protected ref %s" % p)


def _check_git_push(rest, flags, longs, dir_args, cfg, cwd):
    force = "f" in flags or "--force" in longs
    lease = "--force-with-lease" in longs or "--force-if-includes" in longs
    if lease and not cfg.get("allow_force_with_lease"):
        force = True
    mirror = "--mirror" in longs
    delete = "d" in flags or "--delete" in longs
    everything = "--all" in longs or "--branches" in longs

    # positionals: skip option arguments like `-o value`
    pos = []
    skip = False
    for a in rest:
        if skip:
            skip = False
            continue
        if a in ("-o", "--push-option", "--repo", "--receive-pack", "--exec"):
            skip = True
            continue
        if a.startswith("-"):
            continue
        pos.append(a)
    refspecs = pos[1:] if pos else []

    if mirror:
        block(cfg, "git_force_push_protected", "`git push --mirror` force-overwrites every ref on the remote")

    dests = []
    for spec in refspecs:
        spec_force = spec.startswith("+")
        spec = spec.lstrip("+")
        src, _, dst = spec.partition(":")
        dest = dst if dst else src
        if delete or (not dst and spec.startswith(":")) or (dst and not src):
            if branch_protected(dest, cfg):
                block(cfg, "git_delete_protected_branch", "`git push` would delete protected remote branch `%s`" % dest)
            continue
        dests.append((dest, spec_force))

    def resolve(name):
        if name in ("HEAD", "", "@"):
            return current_branch(cwd, dir_args)
        return name

    if force:
        if everything:
            block(cfg, "git_force_push_protected", "`git push --force --all` would overwrite every branch, including protected ones")
        if not dests:
            cur = current_branch(cwd, dir_args)
            if cur is None:
                block(cfg, "git_force_push_unknown_branch",
                      "force-push with no explicit refspec: cannot verify the target branch (name it explicitly, e.g. `git push --force-with-lease origin my-feature`)")
            elif branch_protected(cur, cfg):
                block(cfg, "git_force_push_protected", "force-pushing the current branch `%s`, which is protected" % cur)
        for dest, _ in dests:
            resolved = resolve(dest)
            if resolved is None:
                block(cfg, "git_force_push_unknown_branch", "force-push target `%s` cannot be resolved to a branch name" % dest)
            if branch_protected(resolved, cfg):
                block(cfg, "git_force_push_protected", "force-pushing to protected branch `%s`" % resolved)
    else:
        for dest, spec_force in dests:
            resolved = resolve(dest)
            if spec_force and (resolved is None or branch_protected(resolved, cfg)):
                block(cfg, "git_force_push_protected", "`+%s` is a force-push refspec targeting a protected branch" % dest)
            if cfg.get("block_direct_push_to_protected") and branch_protected(resolved, cfg):
                block(cfg, "git_direct_push_protected", "direct push to protected branch `%s` (open a pull request instead)" % resolved)
        if cfg.get("block_direct_push_to_protected") and not dests:
            cur = current_branch(cwd, dir_args)
            if cur and branch_protected(cur, cfg):
                block(cfg, "git_direct_push_protected", "direct push to protected branch `%s` (open a pull request instead)" % cur)


# --------------------------------------------------------------------------- #
# Rules: databases and SQL
# --------------------------------------------------------------------------- #

SAFE_TEXT_COMMANDS = {"echo", "printf", "grep", "egrep", "fgrep", "rg", "ag",
                      "ack", "git", "cat", "less", "more", "head", "tail",
                      "sed", "awk", "ls", "find", "gh", "test", "[", "wc",
                      "sort", "uniq", "diff", "tr", "cut", "man", "which",
                      "type", "read", "true", "false", "comm", "column",
                      "bat", "jq", "yq", "glow", "mdcat"}
DB_CLIENTS = {"psql", "mysql", "mariadb", "sqlite3", "sqlite", "sqlcmd",
              "mongo", "mongosh", "clickhouse-client", "clickhouse",
              "duckdb", "cockroach", "turso", "wrangler", "prisma",
              "pgcli", "mycli", "litecli", "usql", "bq", "snowsql",
              "dbt", "sqlplus", "redis-cli", "cqlsh", "influx"}
SQL_DROP_RE = re.compile(r"\bDROP\s+(?:TABLE|DATABASE|SCHEMA|COLLECTION)\b", re.I)
SQL_TRUNCATE_RE = re.compile(r"\bTRUNCATE\s+(?:TABLE\s+)?[`\"\[]?[A-Za-z_]", re.I)
SQL_DELETE_ALL_RE = re.compile(
    r"\bDELETE\s+FROM\s+[`\"\[]?[A-Za-z_][A-Za-z0-9_.\"`\]]*\s*(?:;|$|\"|'|\)|--)", re.I | re.M)


def scan_sql(text, cfg, what):
    if SQL_DROP_RE.search(text):
        block(cfg, "sql_drop", "SQL `DROP TABLE/DATABASE/SCHEMA` in %s permanently destroys data" % what)
    if SQL_TRUNCATE_RE.search(text):
        block(cfg, "sql_truncate", "SQL `TRUNCATE` in %s empties a table irreversibly" % what)
    if SQL_DELETE_ALL_RE.search(text):
        block(cfg, "sql_delete_without_where", "SQL `DELETE FROM <table>` without a WHERE clause in %s" % what)


def check_sql_segment(base, args, raw_segment, cfg):
    if base in SAFE_TEXT_COMMANDS or not base:
        return
    scan_sql(raw_segment, cfg, "`%s` command" % base)


def check_db_tools(base, args, meta, cfg, cwd):
    joined = " ".join(args)
    lower = joined.lower()
    if base == "dropdb":
        block(cfg, "db_destroy", "`dropdb` deletes an entire PostgreSQL database")
    if base == "mysqladmin" and "drop" in args:
        block(cfg, "db_destroy", "`mysqladmin drop` deletes an entire MySQL database")
    if base == "redis-cli" and re.search(r"\bflush(all|db)\b", lower):
        block(cfg, "db_destroy", "`redis-cli FLUSHALL/FLUSHDB` wipes Redis data")
    if base in ("mongo", "mongosh") and re.search(r"dropdatabase\s*\(|\.drop\s*\(", lower):
        block(cfg, "db_destroy", "MongoDB dropDatabase()/drop() destroys data")
    if base == "prisma":
        if re.search(r"\bmigrate\s+reset\b", lower) or ("--force-reset" in args):
            block(cfg, "db_destroy", "`prisma migrate reset` / `--force-reset` drops and recreates the database")
    if base in ("rails", "rake", "bin/rails") or (base == "bundle" and args[:2] == ["exec", "rails"]):
        if re.search(r"\bdb:(drop|reset|purge)\b", lower):
            block(cfg, "db_destroy", "Rails `db:drop/db:reset` destroys the database")
    if base == "artisan" or (base in ("php",) and args and base_name(args[0]) == "artisan"):
        if re.search(r"\b(migrate:fresh|migrate:reset|migrate:refresh|db:wipe)\b", lower):
            block(cfg, "db_destroy", "Laravel `migrate:fresh/db:wipe` drops all tables")
    if base.startswith("python") and args and base_name(args[0]) == "manage.py":
        if any(a in ("flush", "reset_db", "sqlflush") for a in args[1:]):
            block(cfg, "db_destroy", "Django `manage.py flush` empties every table")
    if base in ("sequelize", "sequelize-cli") and "db:drop" in args:
        block(cfg, "db_destroy", "`sequelize db:drop` deletes the database")
    if base == "knex" and "migrate:rollback" in args and "--all" in args:
        block(cfg, "db_destroy", "`knex migrate:rollback --all` reverts every migration")
    if base == "supabase" and args[:2] == ["db", "reset"]:
        block(cfg, "db_destroy", "`supabase db reset` wipes the local database")
    if base == "turso" and args[:2] == ["db", "destroy"]:
        block(cfg, "db_destroy", "`turso db destroy` deletes a database")
    if base == "wrangler" and args[:2] == ["d1", "delete"]:
        block(cfg, "db_destroy", "`wrangler d1 delete` deletes a D1 database")
    if base == "pscale" and args[:2] == ["database", "delete"]:
        block(cfg, "db_destroy", "`pscale database delete` deletes a PlanetScale database")
    if base == "heroku" and args and args[0] in ("pg:reset", "apps:destroy"):
        block(cfg, "cloud_destroy", "`heroku %s` is irreversible" % args[0])
    if base in ("flyctl", "fly") and args[:2] in (["apps", "destroy"], ["postgres", "destroy"], ["volumes", "destroy"]):
        block(cfg, "cloud_destroy", "`fly %s %s` destroys infrastructure" % (args[0], args[1]))


def check_cloud_infra(base, args, meta, cfg, cwd):
    joined = " ".join(args)
    if base in ("terraform", "tofu", "pulumi") and args and args[0] == "destroy":
        block(cfg, "cloud_destroy", "`%s destroy` tears down live infrastructure" % base)
    if base == "aws" and len(args) >= 2:
        svc, op = args[0], args[1]
        if svc == "s3" and op == "rb":
            block(cfg, "cloud_destroy", "`aws s3 rb` removes an S3 bucket")
        if svc == "s3" and op == "rm" and "--recursive" in args:
            block(cfg, "cloud_destroy", "`aws s3 rm --recursive` deletes every object under the prefix")
        if op.startswith(("delete-", "terminate-", "deregister-")) and svc in ("rds", "ec2", "dynamodb", "s3api", "lambda", "eks", "ecs", "cloudformation", "iam", "route53", "elasticache", "sqs", "sns"):
            block(cfg, "cloud_destroy", "`aws %s %s` deletes cloud resources" % (svc, op))
    if base == "gcloud" and "delete" in args and not any(a.startswith("--dry-run") for a in args):
        block(cfg, "cloud_destroy", "`gcloud ... delete` deletes cloud resources")
    if base == "gsutil" and args[:1] == ["rm"] and ("-r" in args or "-R" in args):
        block(cfg, "cloud_destroy", "`gsutil rm -r` deletes every object under the prefix")
    if base == "az" and "delete" in args:
        block(cfg, "cloud_destroy", "`az ... delete` deletes cloud resources")
    if base == "kubectl" and args[:1] == ["delete"]:
        if any(a in ("namespace", "ns", "namespaces", "pv", "pvc", "persistentvolume", "persistentvolumeclaim") for a in args[1:3]) or "--all" in args or "--all-namespaces" in args or "-A" in args:
            block(cfg, "cloud_destroy", "`kubectl delete` of namespaces/volumes/--all destroys workloads and data")
    if base == "docker":
        if args[:2] == ["system", "prune"] or args[:2] == ["volume", "prune"] or args[:2] == ["volume", "rm"]:
            block(cfg, "container_volume_destroy", "`docker %s %s` deletes container volumes/data" % (args[0], args[1]))
        if args[:2] == ["compose", "down"] and ("-v" in args or "--volumes" in args):
            block(cfg, "container_volume_destroy", "`docker compose down -v` deletes named volumes (database data)")
    if base == "docker-compose" and args[:1] == ["down"] and ("-v" in args or "--volumes" in args):
        block(cfg, "container_volume_destroy", "`docker-compose down -v` deletes named volumes (database data)")
    if base == "vercel" and args[:1] in (["remove"], ["rm"]):
        block(cfg, "cloud_destroy", "`vercel remove` deletes a deployment/project")


# --------------------------------------------------------------------------- #
# Rules: permissions, disks, system
# --------------------------------------------------------------------------- #

def mode_is_dangerous(mode):
    if re.match(r"^0?[0-7]{3,4}$", mode):
        digits = mode[-3:]
        return digits == "000" or digits[-1] in "67"
    # symbolic: o+w, a+w, +w, a=rwx, o=rwx, ugo+rwx
    parts = mode.split(",")
    for p in parts:
        m = re.match(r"^([ugoa]*)([+=])([rwxXst]*)$", p)
        if not m:
            continue
        who, op, perms = m.groups()
        if "w" in perms and (who == "" or "o" in who or "a" in who):
            return True
    return False


def check_permissions(base, args, meta, cfg, cwd):
    if base not in ("chmod", "chown", "chgrp"):
        return
    flags = short_flags(args)
    recursive = "R" in flags or "--recursive" in args
    pos = positional_args(args)
    if not pos:
        return
    mode, targets = pos[0], pos[1:]
    if base == "chmod":
        if recursive and mode_is_dangerous(mode):
            block(cfg, "chmod_world_writable",
                  "`chmod -R %s` makes an entire tree world-writable/unusable" % mode)
        for t in targets:
            reason = classify_protected_dir(t, cfg, cwd)
            if reason and (recursive or mode_is_dangerous(mode)):
                block(cfg, "chmod_protected", "`chmod %s` on %s" % (mode, reason))
    else:
        if recursive:
            for t in targets:
                reason = classify_protected_dir(t, cfg, cwd)
                if reason:
                    block(cfg, "chown_recursive_protected", "`%s -R` on %s breaks system ownership" % (base, reason))


def check_disks(base, args, meta, cfg, cwd):
    if base.startswith("mkfs") or base in ("mke2fs", "mkswap", "newfs", "newfs_hfs", "newfs_apfs", "newfs_msdos", "wipefs", "sfdisk", "format"):
        block(cfg, "disk_format", "`%s` formats/wipes a disk or partition" % base)
    if base in ("fdisk", "cfdisk", "gdisk", "parted"):
        listing = "-l" in args or "--list" in args or "print" in args
        if not listing:
            block(cfg, "disk_format", "`%s` modifies partition tables" % base)
    if base == "diskutil" and args:
        verb = args[0].lower()
        sub = args[1].lower() if len(args) > 1 else ""
        if verb in ("erasedisk", "erasevolume", "partitiondisk", "zerodisk", "secureerase", "reformat") \
                or (verb == "apfs" and sub in ("deletecontainer", "deletevolume")):
            block(cfg, "disk_format", "`diskutil %s` erases a disk/volume" % " ".join(args[:2] if verb == "apfs" else args[:1]))


def check_system(base, args, meta, cfg, cwd):
    if base in ("shutdown", "reboot", "halt", "poweroff", "telinit"):
        block(cfg, "system_power", "`%s` would power off or restart the machine" % base)
    if base == "init" and args and args[0] in ("0", "6"):
        block(cfg, "system_power", "`init %s` would power off or restart the machine" % args[0])
    if base == "systemctl" and args and args[0] in ("poweroff", "reboot", "halt", "kexec", "suspend", "hibernate"):
        block(cfg, "system_power", "`systemctl %s` would power off or restart the machine" % args[0])
    if base == "kill" and "-1" in args and any(a in ("-9", "-KILL", "-SIGKILL", "-15", "-TERM", "-SIGTERM", "-s") for a in args):
        block(cfg, "kill_all_processes", "`kill -9 -1` kills every process you own, including this session")
    if base == "killall5":
        block(cfg, "kill_all_processes", "`killall5` kills every process")
    if base == "crontab" and "-r" in args:
        block(cfg, "crontab_remove", "`crontab -r` deletes all scheduled jobs without confirmation")
    if base == "launchctl" and args and args[0] in ("bootout", "unload") and any("/System/" in a for a in args):
        block(cfg, "system_power", "unloading system launchd services")
    if base in ("iptables", "ip6tables", "nft") and ("-F" in args or "--flush" in args or args[:2] == ["flush", "ruleset"]):
        block(cfg, "firewall_flush", "flushing firewall rules")
    if base == "history" and "-c" in args:
        block(cfg, "history_clear", "`history -c` erases the shell history")


# --------------------------------------------------------------------------- #
# Rules: remote code piping and fork bombs (raw text)
# --------------------------------------------------------------------------- #

FETCHERS = {"curl", "wget", "fetch", "http", "https", "aria2c", "httpie"}
SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish", "csh", "tcsh", "ash", "mksh"}
INTERPRETERS = {"python", "python2", "python3", "perl", "ruby", "node", "php", "deno", "bun", "lua", "pwsh", "powershell"}
FORK_BOMB_RE = re.compile(r"(\w+|:)\s*\(\s*\)\s*\{[^}]*\1\s*\|\s*\1\s*&[^}]*\}\s*;?\s*\1")
PROC_SUBST_SHELL_RE = re.compile(
    r"(?:^|[\s;&|(])(?:sudo\s+(?:-\S+\s+)*)?(?:source|\.|(?:ba|z|da|k|fi)?sh|python3?|perl|ruby|node)\s+(?:-\S+\s+)*<\(\s*(?:curl|wget|fetch)\b")
CMD_SUBST_SHELL_RE = re.compile(
    r"(?:^|[\s;&|(])(?:sudo\s+(?:-\S+\s+)*)?(?:eval|source|\.|(?:ba|z|da|k)?sh\s+(?:-\S+\s+)*-c)\s+[\"']?\s*(?:\$\(|`)\s*(?:curl|wget|fetch)\b")


def interpreter_reads_stdin(base, args):
    """Does `base args` execute a program taken from stdin?"""
    if base in SHELLS:
        # bash -c 'x' does not read stdin; bash / bash -s / bash - / bash -s -- args do
        return "-c" not in args and not any(a.startswith("-") and "c" in a[1:] and not a.startswith("--") for a in args) and (
            not positional_args(args) or "-s" in args or "-" in args)
    if base in INTERPRETERS:
        if base.startswith("python"):
            return "-c" not in args and "-m" not in args and (not positional_args(args) or args[:1] == ["-"])
        if base == "perl":
            return "-e" not in args and "-E" not in args and (not positional_args(args) or args[:1] == ["-"])
        if base == "ruby":
            return "-e" not in args and (not positional_args(args) or args[:1] == ["-"])
        if base in ("node", "deno", "bun"):
            return "-e" not in args and "--eval" not in args and "-p" not in args and (not positional_args(args) or args[:1] == ["-"])
        if base == "php":
            return not positional_args(args) or args[:1] == ["--"]
        return not positional_args(args)
    return base in ("eval", "source", ".")


def check_pipe_to_shell(analysed_segments, cfg):
    """analysed_segments: list of (connector, base, args, meta)."""
    prev_base = None
    for connector, base, args, meta in analysed_segments:
        if connector == "|" and prev_base in FETCHERS and (base in SHELLS or base in INTERPRETERS or base in ("eval", "source", ".")):
            if interpreter_reads_stdin(base, args):
                block(cfg, "pipe_to_shell",
                      "piping `%s` output straight into `%s` executes unreviewed remote code (download to a file, inspect it, then run)" % (prev_base, base))
        prev_base = base if base else prev_base


def check_raw(cmd, cfg):
    if FORK_BOMB_RE.search(cmd):
        block(cfg, "fork_bomb", "fork bomb pattern detected")
    if PROC_SUBST_SHELL_RE.search(cmd) or CMD_SUBST_SHELL_RE.search(cmd):
        block(cfg, "pipe_to_shell", "executing remote content fetched with curl/wget via process/command substitution")
    for pat in cfg.get("block_patterns", []):
        try:
            if re.search(pat, cmd):
                block(cfg, "block_pattern", "matches configured block pattern %r" % pat)
        except re.error:
            continue


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

NESTED_SHELLS = SHELLS | {"su", "script"}


def nested_command_strings(base, args):
    """Return command strings embedded in `bash -c ...`, `eval ...`, `su -c`, etc."""
    out = []
    if base in NESTED_SHELLS:
        for i, a in enumerate(args):
            takes_cmd = a == "-c" or (a.startswith("-") and not a.startswith("--") and a.endswith("c") and len(a) > 1)
            if takes_cmd and i + 1 < len(args):
                out.append(args[i + 1])
                break
    elif base == "eval":
        out.append(" ".join(args))
    elif base in ("ssh",):
        pass  # remote host: out of scope
    elif base == "find":
        # find ... -exec <cmd> ... ; -> analyse the exec'd command
        for i, a in enumerate(args):
            if a in ("-exec", "-execdir", "-ok", "-okdir"):
                inner = []
                for t in args[i + 1:]:
                    if t in (";", "+"):
                        break
                    inner.append(t)
                if inner:
                    out.append(" ".join(shlex.quote(t) for t in inner))
    elif base in ("watch", "timeout"):
        pass  # handled as wrappers
    elif base in ("parallel", "xargs"):
        pass
    return out


def analyse(cmd, cfg, cwd, depth=0):
    """Raise Block if anything in `cmd` is destructive."""
    if depth > MAX_RECURSION_DEPTH or not cmd or not cmd.strip():
        return
    check_raw(cmd, cfg)
    cmd, heredocs = preprocess(cmd)

    for inner in extract_substitutions(cmd):
        analyse(inner, cfg, cwd, depth + 1)

    analysed = []
    for connector, seg in split_segments(cmd):
        tokens = tokenize(seg)
        tokens, redirects = strip_redirections(tokens)
        base, args, meta = strip_wrappers(tokens)
        analysed.append((connector, base, args, meta))
        if not base:
            # e.g. `> file` alone, or `: > file`
            check_truncation(base, args, redirects, meta, cfg, cwd)
            continue
        if base == ":":
            check_truncation(base, args, redirects, meta, cfg, cwd)
            continue

        check_rm(base, args, meta, cfg, cwd)
        check_truncation(base, args, redirects, meta, cfg, cwd)
        check_find(base, args, meta, cfg, cwd)
        check_git(base, args, meta, cfg, cwd)
        check_sql_segment(base, args, seg, cfg)
        check_db_tools(base, args, meta, cfg, cwd)
        check_cloud_infra(base, args, meta, cfg, cwd)
        check_permissions(base, args, meta, cfg, cwd)
        check_disks(base, args, meta, cfg, cwd)
        check_system(base, args, meta, cfg, cwd)

        for nested in nested_command_strings(base, args):
            analyse(nested, cfg, cwd, depth + 1)

    check_pipe_to_shell(analysed, cfg)

    # Heredoc bodies fed to a database client are SQL to be inspected.
    for owner_line, body in heredocs:
        owner_bases = set()
        for _, seg in split_segments(owner_line):
            b, _, _ = strip_wrappers(strip_redirections(tokenize(seg))[0])
            owner_bases.add(b)
        if owner_bases & DB_CLIENTS or owner_bases - SAFE_TEXT_COMMANDS - {""} and not owner_bases & {"cat", "tee"}:
            scan_sql(body, cfg, "heredoc")


def allowlisted(cmd, cfg):
    for pat in cfg.get("allow_patterns", []):
        try:
            if re.search(pat, cmd):
                return True
        except re.error:
            continue
    return False


def format_block_message(cmd, err):
    shown = cmd if len(cmd) <= 300 else cmd[:297] + "..."
    shown = shown.replace("\n", "\\n")
    return (
        "BLOCKED by %s (rule: %s)\n"
        "  reason : %s\n"
        "  command: %s\n"
        "  This command was not executed. If it is genuinely required, ask the user to run it\n"
        "  manually, or add a regex to \"allow_patterns\" (or the rule id to \"disabled_rules\")\n"
        "  under \"destructive\" in .claude/guardrails.json.\n"
        % (HOOK_NAME, err.rule, err.reason, shown)
    )


def main():
    raw = ""
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            raise ValueError("payload is not an object")
    except (ValueError, OSError) as exc:
        cfg = load_config(None)
        if cfg.get("fail_closed"):
            sys.stderr.write("BLOCKED by %s: could not parse hook input (%s)\n" % (HOOK_NAME, exc))
            return 2
        sys.stderr.write("%s: warning: could not parse hook input (%s); allowing\n" % (HOOK_NAME, exc))
        return 0

    tool_name = payload.get("tool_name", "")
    if tool_name != "Bash":
        return 0
    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if not isinstance(command, str) or not command.strip():
        return 0
    cwd = payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()

    cfg = load_config(cwd)
    if allowlisted(command, cfg):
        return 0
    try:
        analyse(command, cfg, cwd)
    except Block as err:
        sys.stderr.write(format_block_message(command, err))
        return 2
    except Exception as exc:  # never crash Claude Code because of the guard
        if cfg.get("fail_closed"):
            sys.stderr.write("BLOCKED by %s: internal error (%s: %s)\n" % (HOOK_NAME, type(exc).__name__, exc))
            return 2
        sys.stderr.write("%s: warning: internal error (%s: %s); allowing\n" % (HOOK_NAME, type(exc).__name__, exc))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
