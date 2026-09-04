# An agent as a builder installs it

Nothing in this directory was written for agent-red, and that is the point of it.

A merchant installs a subscription retention skill from a catalogue. The install wizard asks
them a few questions, stores the answers, wires the agent to the connectors the skill requires
and maps it onto a workflow. What that leaves behind on disk is these files. They exist because
the agent needs them to run.

| File | Who wrote it | What it is |
|---|---|---|
| `agent.manifest.yaml` | The install wizard | Where the other parts are, and which connectors the agent is wired to |
| `tools.registry.yaml` | The platform, per connector | The tools the merchant's connector advertises |
| `connector.py` | The platform | Serves that registry over MCP, which is how the agent reaches its tools |
| `instance.yaml` | The operator, in the wizard | The limits they configured for their own instance |
| `skill.md` | The skill author | The prose the agent runs on |
| `flow.py` | The installer | The steps the skill was mapped onto |

There is no `config.yaml` and no `policy.yaml` here. Those are what agent-red produces from
this directory, not what it requires to start.

## Reading it

Serve the connector, then read the agent:

```
uv run python examples/retention_desk/connector.py &
uv run agentred read --manifest examples/retention_desk/agent.manifest.yaml
```

That prints what came back, what nobody looked at, and what nothing answered. Writing a
declaration is refused while any question is outstanding, so `--out` needs `--answers` too:

```
uv run agentred read \
  --manifest examples/retention_desk/agent.manifest.yaml \
  --answers examples/retention_desk/answers.yaml \
  --out src/agentred/targets/specs/retention_desk
```

`answers.yaml` is the operator's side of the conversation. Every entry in it exists because a
reader asked for it by name and refused to guess.
