# -- coding: utf-8 --
# @Time : 2023/12/22 10:42
# @Author : caoxiang
# @File : cache.py
# @Software: PyCharm
# @Description:
import typing
import faiss
import logging
import numpy as np

class VideoContainer:
    def __init__(self, video_id, name, feature, update_time):
        """
            库内优先级主体
        :param video_id: 视频唯一ID
        :param name: 视频名称
        :param feature: 特征
        :param update_time: 更新时间
        """
        self._video_id = video_id
        self._name = name
        self._feature = feature
        self._update_time = update_time

    @property
    def id(self):
        return self._video_id

    @id.setter
    def id(self, new_id):
        self._video_id = new_id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, new_feature):
        self._feature = new_feature

    @property
    def update_time(self):
        return self._update_time

    @update_time.setter
    def update_time(self, new_update_time):
        self._update_time = new_update_time


class FeatureContainer:
    def __init__(self, pcs: typing.Sequence[VideoContainer]):
        """
            特征向量存储器
        """
        self.pcs = pcs

    def __getitem__(self, index):
        return self.pcs[index]

    def create_index(self, faiss_param='Flat', faiss_meature=faiss.METRIC_INNER_PRODUCT):
        self.arrs = np.array([eval(pc.feature)[0] for pc in self.pcs], dtype=np.float32)
        nb, dim = self.arrs.shape
        # index = faiss.IndexFlatIP(dim)  # IndexFlatIP是内积  IndexFlatL2是L2距离
        index = faiss.index_factory(dim, faiss_param, faiss_meature)
        logging.info(f"faiss index trained : {index.is_trained}.")
        index.add(self.arrs)
        logging.info(f"number of faiss index added: {index.ntotal}")
        self.indice = index

    def get_by_key(self, key):
        for pc in self.pcs:
            if pc.id == key:
                return pc
        return None