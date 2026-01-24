FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    ninja-build \
    autoconf \
    automake \
    libtool \
    pkg-config \
    # Python
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    # Version control
    git \
    # Editor
    emacs \
    # SSH
    openssh-client \
    # Useful utilities
    curl \
    wget \
    ca-certificates \
    gnupg \
    lsb-release \
    vim \
    less \
    htop \
    tree \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for development
ARG USERNAME=dev
ARG USER_UID=1001
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Set up SSH directory with correct permissions
RUN mkdir -p /home/$USERNAME/.ssh \
    && chmod 700 /home/$USERNAME/.ssh \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME/.ssh

# Switch to non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Default command
CMD ["/bin/bash"]