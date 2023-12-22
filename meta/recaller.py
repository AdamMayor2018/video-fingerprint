# -- coding: utf-8 --
# @Time : 2023/12/22 10:55
# @Author : caoxiang
# @File : recaller.py
# @Software: PyCharm
# @Description:
from abc import ABCMeta, abstractmethod
import typing
import logging

import torch

from database.redis import get_redis
import numpy as np
from config.conf_loader import YamlConfigLoader
from util.cache import VideoContainer, FeatureContainer
import hashlib
from meta.extractor import BaseExtractor, TowHeeVideoExtractor
import datetime
import os


class BaseFeatureRecaller(metaclass=ABCMeta):
    @abstractmethod
    def load_to_memory(self):
        pass

    @abstractmethod
    def register_from_directory(self, register_path: str):
        pass

    @abstractmethod
    def retrive_nearest(self, key: np.ndarray):
        pass


class RedisFeatureRecaller(BaseFeatureRecaller):
    def __init__(self, loader: YamlConfigLoader, extractor: BaseExtractor, auto_loading=True):
        try:
            self.r = get_redis(loader)
            print(self.r)
        except:
            raise Exception("redis connecting error.")
        self.extractor = extractor
        self.load_to_memory() if auto_loading else None
        self.hit_database_ratio = loader.attempt_load_param("hit_database_ratio")

    def load_to_memory(self):
        """
            从redis中加载全部特征向量到内存数据结构中
        :return:
        """
        keys = self.r.keys()
        pcs = []
        for i, k in enumerate(keys):
            all = self.r.hgetall(k)
            pc = VideoContainer(video_id=k, **all)
            pcs.append(pc)
        self.fc = FeatureContainer(pcs)
        #  建立索引
        self.fc.create_index()

    @torch.no_grad()
    def register_from_directory(self, register_directory):
        """
        从目录中批量加载视频，并将特征注册到redis中， 文件夹中是需要去重的视频集合
        :return:
        """
        for i, video_name in enumerate(os.listdir(register_directory)):
            video_path = os.path.join(register_directory, video_name)
            emb = self.extractor.extract_video(video_path)
            if len(emb.shape) == 1:
                emb = emb.reshape(1, -1)
            emb = emb / np.linalg.norm(emb, axis=1)
            emb = emb.astype('float32')
            # 随机生成十六位哈希值作为key
            key = hashlib.sha1(f"{video_path}".encode()).hexdigest()
            self.r.hset(key, mapping={"name": video_name.split(".")[0] if "." in video_name else video_name,
                                      "feature": str(emb.tolist()),
                                      "update_time": datetime.datetime.now().strftime("%Y-%m-%d %X"),
                                      })

    def retrive_nearest(self, key: np.ndarray, topk: int = 1):
        """
            召回特征相似最高的库内目标
        :param topk: 取最接近的topk
        :param key: 用于查询的向量
        :return:
        """
        if len(key.shape) == 1:
            key = key.reshape(1, -1)
        if self.fc:
            key = key / np.linalg.norm(key, axis=1)
            key = key.astype('float32')
            # embeddings = self.fc.arrs
            # assert embeddings.shape[1] == key.shape[1], "Embdding element size not equals to key size!"
            # l2dist = np.dot(key, embeddings.T)
            D, I = self.fc.indice.search(key, topk)
            indices = I[0].tolist()
            sims = D[0]
            # ind = int(np.argmax(l2dist, axis=1))
            # max_sim = np.max(l2dist, axis=1)

            return indices, sims

        else:
            raise Exception("Embdding Memory container is empty.Consider load first!")

    def recall_item(self, video_path: str, topk: int = 1):
        key_feature = self.extractor.extract_video(video_path)
        indices, sims = self.retrive_nearest(key_feature, topk)
        result = []
        for (ind, max_sim) in zip(indices, sims):
            vc = self.fc[ind]
            result.append({"video_id": vc.id, "video_name": vc.name, "similarity": max_sim})
        return result


if __name__ == '__main__':
    path = "/data/cx/ysp/video-fingerprint/notebooks/VCDB_core_sample/troy_achilles_and_hector"
    conf_loader = YamlConfigLoader(yaml_path="../config/general_config.yaml")
    print(conf_loader.config_dict)
    extractor = TowHeeVideoExtractor(config_loader=conf_loader)
    recaller = RedisFeatureRecaller(loader=conf_loader, extractor=extractor, auto_loading=True)
    #recaller.register_from_directory(path)
    result = recaller.recall_item("/data/cx/ysp/video-fingerprint/notebooks/VCDB_core_sample/scent_of_woman_tango/f3c8c0c9b93e0a49d2508eee4aae618c1d69e082.flv", topk=5)
    #result = recaller.recall_item("/data/cx/ysp/video-fingerprint/notebooks/VCDB_core_sample/troy_achilles_and_hector/ee417a6b882853ffcd3f78b380b0205a9411f4d6.flv", topk=5)
    print(result)
