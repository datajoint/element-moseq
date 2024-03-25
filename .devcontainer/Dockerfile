FROM python:3.9-slim@sha256:5f0192a4f58a6ce99f732fe05e3b3d00f12ae62e183886bca3ebe3d202686c7f

ENV PATH /usr/local/bin:$PATH
ENV PYTHON_VERSION 3.9.17

RUN \
    adduser --system --disabled-password --shell /bin/bash vscode && \
    # install docker
    apt-get update && \
    apt-get install ca-certificates curl gnupg lsb-release -y && \
    mkdir -m 0755 -p /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y && \
    usermod -aG docker vscode && \
    apt-get clean

RUN \
    # dev setup
    apt update && \
    apt-get install sudo git bash-completion graphviz default-mysql-client s3fs procps -y && \
    usermod -aG sudo vscode && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    pip install --no-cache-dir --upgrade black pip nbconvert && \
    echo '. /etc/bash_completion' >> /home/vscode/.bashrc && \
    echo 'export PS1="\[\e[32;1m\]\u\[\e[m\]@\[\e[34;1m\]\H\[\e[m\]:\[\e[33;1m\]\w\[\e[m\]$ "' >> /home/vscode/.bashrc && \
    apt-get clean

COPY ./ /tmp/element-moseq/

RUN \
    # pipeline dependencies
    apt-get install gcc g++ ffmpeg libsm6 libxext6 libgl1 libegl1 -y && \
    pip install --no-cache-dir -e /tmp/element-moseq[elements,tests] && \
    # clean up
    rm -rf /tmp/element-moseq/ && \
    apt-get clean

# Install Keypoint-MoSeq (CPU version)
RUN pip install "jax[cpu]==0.3.22" -f https://storage.googleapis.com/jax-releases/jax_releases.html

ENV DJ_HOST fakeservices.datajoint.io
ENV DJ_USER root
ENV DJ_PASS simple

ENV DATA_MOUNTPOINT /workspaces/element-moseq/example_data
ENV KPMS_ROOT_DATA_DIR $DATA_MOUNTPOINT/inbox
ENV KPMS_PROCESSED_DATA_DIR $DATA_MOUNTPOINT/outbox
ENV DATABASE_PREFIX neuro_

USER vscode
CMD bash -c "sudo rm /var/run/docker.pid; sudo dockerd"

ENV LD_LIBRARY_PATH="/lib:/opt/conda/lib"