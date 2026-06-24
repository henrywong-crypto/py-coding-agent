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

## LSP (optional)

```sh
npm i -g pyright                    # for lsp_python
rustup component add rust-analyzer  # for lsp_rust
```

## Extensions

`agent.py` ships the core (tools, session, UI, model wiring). Language support
lives in sibling files — `python.py`, `rust.py` — each exposing a `HOOKS`
tuple. Any `.py` beside `agent.py` with a `HOOKS` tuple is auto-loaded at
startup; drop one in to add support, delete one to remove it. Extensions
import what they need (`Tool`, `LspServer`, `_install_lsp`, …) from `agent`.
