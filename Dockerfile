# ============================================================
# JaxMARL-HFT Docker Image
# ============================================================
# NOTE: This image is x86_64/amd64 only (NVIDIA base image).
#       It will NOT work on ARM64 systems (e.g. Apple Silicon, AWS Graviton).
#
# Build:  make build
# Run:    make run            (opens interactive shell)
#         make ppo_2player gpu=0   (runs 2-player training on GPU 0)
#
# The Makefile mounts:
#   - The repo          -> /home/myuser/
#   - Your data dir     -> /home/myuser/data/
#   - Your scratch dir  -> /home/myuser/scratch/
#
# So inside the container, the default env config paths
# (alphatradePath="/home/myuser", dataPath="/home/myuser/data")
# work without modification.
#
# For WandB logging, set your API key before running:
#   export WANDB_API_KEY=<your-key>
#   make ppo_2player gpu=0
# Or pass it via docker run:
#   docker run -e WANDB_API_KEY=<your-key> ...
# ============================================================

FROM nvcr.io/nvidia/jax:25.10-py3
# Create user
ARG UID
ARG MYUSER
RUN useradd -m -u $UID --create-home ${MYUSER} && \
    echo "${MYUSER}:${MYUSER}" | chpasswd && \
    adduser ${MYUSER} sudo && \
    chown -R ${MYUSER}:${MYUSER} /home/${MYUSER}

# default workdir
WORKDIR /home/${MYUSER}/
COPY --chown=${MYUSER} --chmod=777 . .

# Install system packages
RUN apt-get update && \
    apt-get install -y tmux && \
    apt-get install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
    libgl1 \
    p7zip-full \
    unrar \
    htop \
    graphviz

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Install Go (used for WandB internals)
RUN wget https://go.dev/dl/go1.24.5.linux-amd64.tar.gz
RUN rm -rf /usr/local/go
RUN tar -C /usr/local -xzf go1.24.5.linux-amd64.tar.gz
ENV PATH="$PATH:/usr/local/go/bin"

USER ${MYUSER}

# JAX memory settings — prevent OOM by disabling full preallocation
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Ensure repo root is on the Python path
ENV PYTHONPATH="/home/${MYUSER}:$PYTHONPATH"

# WandB credentials — pass API key via environment variable when running:
# e.g. docker run -e WANDB_API_KEY=<key> ...
RUN git config --global --add safe.directory /home/${MYUSER}
