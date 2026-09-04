# One image, every process. Which agent it serves, or whether it serves the tool server
# instead, is a command argument, so the containers in docker-compose.yml differ only by their
# spec directory, their entrypoint and their ports.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AGENTRED_TARGET_MODE=test

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir . && pip install --no-cache-dir uvicorn

EXPOSE 8081

# The Agent SDK needs the Claude Code CLI on PATH to run an agent.
RUN apt-get update \
 && apt-get install -y --no-install-recommends nodejs npm \
 && npm install -g @anthropic-ai/claude-code \
 && apt-get purge -y npm \
 && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["python", "-m", "agentred.targets.serve"]
