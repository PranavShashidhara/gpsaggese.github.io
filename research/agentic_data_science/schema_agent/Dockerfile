FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:$PATH"

# This allows 'import helpers' to work if helpers is inside /git_root/helpers_root
ENV PYTHONPATH="/git_root/research/agentic_data_science/schema_agent:/git_root/helpers_root:${PYTHONPATH:-}"

RUN apt-get update && apt-get install -y \
    ca-certificates build-essential curl sudo gnupg git vim \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

RUN uv venv /opt/venv

# Requirements installation
COPY requirements.txt /install/requirements.txt
RUN uv pip install --python /opt/venv/bin/python --no-cache -r /install/requirements.txt jupyterlab

# Create the skeleton directory structure
WORKDIR /git_root

# Address reviewer feedback: We assume schema_agent.py is in the context
# We will chmod it inside the container during build or via the mount script
EXPOSE 8888

CMD ["/bin/bash"]