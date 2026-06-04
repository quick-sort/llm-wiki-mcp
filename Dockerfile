FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Node.js (required by Claude Code CLI) and gh CLI
ENV FNM_DIR="/opt/fnm"
ENV PATH="/opt/fnm/aliases/default/bin:$PATH"
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git wget unzip \
    && curl -fsSL https://fnm.vercel.app/install | bash -s -- --install-dir /usr/local/bin --skip-shell \
    && fnm install 24 \
    && fnm default 24 \
    && chmod -R 755 /opt/fnm/aliases/default/bin \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 999 agent \
    && useradd --uid 1000 --gid 999 -m agent \
    && echo "registry=https://registry.npmmirror.com" > /home/agent/.npmrc \
    && npm config set prefix '/home/agent/.npm-global' \
    && npm install -g @anthropic-ai/claude-code@2.1.110 \
    && npm cache clean --force

# Clone llm-wiki-agent skills at build time
RUN git clone --depth 1 https://github.com/SamurAIGPT/llm-wiki-agent.git /app/skills/llm-wiki-agent \
    && chown -R agent:agent /app/skills

VOLUME ["/home/agent"]

WORKDIR /app
ENV UV_LINK_MODE=copy
ENV HOME="/home/agent"

# Install Python dependencies (cached layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY pyproject.toml uv.lock /app/
COPY src /app/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

RUN chown -R agent:agent /app /home/agent

USER agent

ENV UV_CACHE_DIR="/home/agent/.cache/uv"
ENV PATH="/home/agent/.npm-global/bin:$PATH"
ENV ANTHROPIC_API_KEY=""
ENV LLM_WIKI_ROOT=/data/wikis

VOLUME /data

EXPOSE 8080

CMD ["uv", "run", "llm-wiki-mcp", "--wikis-root", "/data/wikis"]
