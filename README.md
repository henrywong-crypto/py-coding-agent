# Python coding agent

## New / continue

```sh
uv run agent.py "write a hello world in rust and run it"    # continues the most recent session
uv run agent.py --new "explain rust lifetimes"              # starts a new session
```

Use `--new` only when prior sessions exist and you want to start over.

## Sessions

```sh
uv run agent.py --list-sessions                             # list prior sessions and exit
uv run agent.py --session PATH "..."                        # opens a specific session file
```
