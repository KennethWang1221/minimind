uv pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
git clone https://huggingface.co/jingyaogong/MiniMind2 # or https://www.modelscope.cn/models/gongjy/MiniMind2
git clone https://www.modelscope.cn/datasets/gongjy/minimind_dataset.git
git clone https://www.modelscope.cn/gongjy/MiniMind2-PyTorch.git
cd out 
ln -s ../MiniMind2-PyTorch/* . 
cd ../