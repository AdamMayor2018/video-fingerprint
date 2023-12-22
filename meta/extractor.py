# -- coding: utf-8 --
# @Time : 2023/12/22 11:12
# @Author : caoxiang
# @File : extractor.py
# @Software: PyCharm
# @Description:
import typing
from abc import ABCMeta, abstractmethod
from config.conf_loader import YamlConfigLoader
from towhee import pipe
from meta.distill_and_select import DistillAndSelect
from meta.video_decoder import VideoDecoder

class BaseExtractor(metaclass=ABCMeta):
    @abstractmethod
    def extract_video(self, video_path: str):
        pass


class TowHeeVideoExtractor(BaseExtractor):
    def __init__(self, config_loader: YamlConfigLoader):
        super(BaseExtractor, self).__init__()
        self.model = None
        self.config_loader = config_loader
        self.dns_weight_path = config_loader.attempt_load_param("dns_weight_path")
        self.device = config_loader.attempt_load_param("device")
        self.start = config_loader.attempt_load_param("start")
        self.end = config_loader.attempt_load_param("end")
        self.time_step = config_loader.attempt_load_param("time_step")
        self.pipe = (
            pipe.input('video_path').map('video_path', 'video_gen',
                                         VideoDecoder(start_time=self.start, end_time=self.end, sample_type='time_step_sample',
                                             args={'time_step': self.time_step})).map('video_gen', 'video_list', lambda x: [y for y in x]).map('video_list', 'vec', DistillAndSelect(model_name='cg_student', device=self.device)).output('vec')
        )

    def extract_video(self, video_path: str):
        return self.pipe(video_path).get()[0]


if __name__ == '__main__':
    conf_loader = YamlConfigLoader("../config/general_config.yaml")
    print(conf_loader.config_dict)

    towhee_extractor = TowHeeVideoExtractor(conf_loader)
    print(towhee_extractor.extract_video("../demo_video.flv"))
