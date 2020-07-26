# TFの公式imageをベースにする
# 参考: https://qiita.com/penpenta/items/3b7a0f1e27bbab56a95f
FROM tensorflow/tensorflow:1.15.2-py3-jupyter
USER root
# コンテナを立ち上げるときにマウントするフォルダ経由でコピーする
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y sudo \
    && apt-get install -y lsb-release \
    && apt-get install -y vim \
    && apt-get install -y jq \
    && apt-get install -y tree \
    && apt-get install -y libsm6 libxext6 libxrender-dev \
    && apt-get install -y curl \
    && pip3 install --upgrade pip
# USER $NB_UID
# 作業するディレクトリを変更
WORKDIR /home/uchide
# 予めおいておいたrequirements.txtをインストール
COPY requirements.txt ${PWD}
RUN pip3 install -r requirements.txt
# Cloud SDKをインストール
# インストール後にinit処理が必要
# https://cloud.google.com/sdk/docs/downloads-apt-get
# tensorflowをpullしたので既に入っている説ある
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN sudo apt-get install -y apt-transport-https ca-certificates gnupg
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN sudo apt-get update && sudo apt-get install -y google-cloud-sdk

# plotlyを使うため、Node.jsをインストール
# https://github.com/nodesource/distributions/blob/master/README.md
RUN curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash - \
    && sudo apt-get install -y nodejs
# JupyterLabでplotlyをサポートする, バージョンは適宜サイトを参照してアップデートする
# https://github.com/plotly/plotly.py
ENV NODE_OPTIONS=--max-old-space-size=4096
RUN jupyter labextension install jupyterlab-plotly@4.8.1 --no-build \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.8.1 --no-buil \
    && jupyter lab build
ENV NODE_OPTIONS=
# jupyterlab-tensorboardをインストールする
RUN jupyter labextension install jupyterlab_tensorboard