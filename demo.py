# -- coding: utf-8 --
# @Time : 2023/12/21 15:08
# @Author : caoxiang
# @File : demo.py
# @Software: PyCharm
# @Description:
from towhee import pipe, ops, DataCollection
from meta.distill_and_select import DistillAndSelect


p = (
    pipe.input('video_path') \
        .map('video_path', 'video_gen', ops.video_decode.ffmpeg(start_time=0.0, end_time=1000.0, sample_type='time_step_sample', args={'time_step': 1})) \
        .map('video_gen', 'video_list', lambda x: [y for y in x]) \
        .map('video_list', 'vec',  DistillAndSelect(model_name='fg_bin_student', device='cuda:0')) \
        .output('vec')
)

DataCollection(p('./demo_video.flv')).show()