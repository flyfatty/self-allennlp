# @Time : 2020/12/21 21:25
# @Author : LiuBin
# @File : data_split.py
# @Description : 
# @Software: PyCharm
"""
切分數據為 訓練集、驗證集、測試集
python data_split.py file_path_to_split
"""
import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split


def save(df, fp):
    print(f'{fp}: {df.shape}')
    df.to_csv(fp, index=False, header=False, sep='\t')


if __name__ == '__main__':
    file_path = "/home/liubin/tutorials/pytorch/self-allennlp/data/chatbot/dialogs.txt"
    filepath = sys.argv[1] if len(sys.argv) > 1 else file_path

    if not (os.path.exists(filepath) and os.path.isfile(filepath)):
        raise FileNotFoundError("Exception: 文件不存在")
    basename = os.path.basename(filepath).rsplit('.', 1)[0]
    out_path = os.path.join(os.path.dirname(filepath), basename)
    print("创建输出件夾 {}".format(out_path))
    os.makedirs(out_path, exist_ok=True)

    sep = sys.argv[2] if len(sys.argv) > 2 else '\t'
    df = pd.read_csv(filepath, sep=sep)
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    train, valid = train_test_split(train, test_size=0.2, random_state=42)

    print(f'Saving to {out_path}.')

    save(train, os.path.join(out_path, 'train.tsv'))
    save(valid, os.path.join(out_path, 'valid.tsv'))
    save(test, os.path.join(out_path, 'test.tsv'))
