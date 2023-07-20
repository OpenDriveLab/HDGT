# Installation
HDGT was developed under a certain version of [DGL](https://www.dgl.ai/) after which DGL has made a major refactor. Thus, we suggest using exactly the same environment as provided below to avoid any issues:

```shell
conda create -n hdgt python=3.8
conda activate hdgt

pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install tensorflow tensorboard

wget https://anaconda.org/dglteam/dgl-cuda11.3/0.7.2/download/linux-64/dgl-cuda11.3-0.7.2-py38_0.tar.bz2
conda install --use-local dgl-cuda11.3-0.7.2-py38_0.tar.bz2 -y
conda install protobuf=3.20 -y

pip install -r requirements.txt
```