FROM python:3.10-slim-bookworm

# ヘッドレスの PyBullet（EGL / TinyRenderer）に必要な共有ライブラリ
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    libgl1-mesa-dri \
    fonts-ipafont-gothic \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ビルドコンテキストはリポジトリルート（cueworld パッケージの親）。
# import cueworld が通るよう、パッケージ名のディレクトリに置く。
# subtasks_for_cueworld は同じ /opt/src に compose がマウントする。
WORKDIR /opt/src
COPY cueworld /opt/src/cueworld
ENV PYTHONPATH=/opt/src
WORKDIR /opt/src

CMD ["sleep", "infinity"]
