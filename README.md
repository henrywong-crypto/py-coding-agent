# Python coding agent

```sh
uv run agent.py "write a hello world in rust and run it"    # continues the most recent session
uv run agent.py --new "explain rust lifetimes"              # starts a new session
uv run agent.py --session PATH "..."                        # opens a specific session file
```

Use `--new` only when prior sessions exist and you want to start over.
