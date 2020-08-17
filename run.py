# Author: dgm
# Description: 项目运行入口
# Date: 2020-08-14
import os
import pickle
import argparse

from utils import make_path
from train_val_test import train, test, demo
from models.ALBERT_BILSTM_CRF import Config, AlbertBiLstmCrf
from data_helper import load_sentences, tag_mapping, prepare_dataset, BatchManager

# 参数
parser = argparse.ArgumentParser('ALBET-BiLSTM-CRF')
parser.add_argument('--mode', type=str, help='train eval or demo')
args = parser.parse_args()


if __name__ == "__main__":
    mode = args.mode
    config = Config()

    # 读取数据
    train_sentences = load_sentences(config.train_path, config.lower, config.zeros)
    dev_sentences = load_sentences(config.dev_path, config.lower, config.zeros)
    test_sentences = load_sentences(config.test_path, config.lower, config.zeros)

    # 构建字典
    # tags dict
    if not os.path.isfile(config.map_file):
        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(config.map_file, "wb") as f:
            pickle.dump([tag_to_id, id_to_tag], f)
    else:
        with open(config.map_file, "rb") as f:
            tag_to_id, id_to_tag = pickle.load(f)

    config.num_tags = len(tag_to_id)

    # 构建数据集（训练集、验证集和测试集）
    train_data = prepare_dataset(
        train_sentences, config.max_seq_len, tag_to_id, config.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, config.max_seq_len, tag_to_id, config.lower
    )
    test_data = prepare_dataset(
        test_sentences, config.max_seq_len, tag_to_id, config.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), 0, len(test_data)))

    train_manager = BatchManager(train_data, config.batch_size)
    dev_manager = BatchManager(dev_data, config.batch_size)
    test_manager = BatchManager(test_data, config.batch_size)

    # 构建模型
    model = AlbertBiLstmCrf(config)
    make_path(config)

    # 训练模型
    if mode == "train":
        train(model, config, train_manager, dev_manager, id_to_tag)
    elif mode == "test":
        test(model, config, test_manager, id_to_tag)
    else:
        demo(model, config, id_to_tag, tag_to_id)

