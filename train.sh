pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboardX
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torchvision
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/yufeng/disen_rob/data
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/yufeng/disen_rob/rst_adv
python3 disentangle_training.py --parallel --batch_size 3000
