import logging
import av
import numpy as np
from towhee.operator.base import PyOperator

from .cpu_decode import PyAVDecode

logger = logging.getLogger()


class SAMPLE_TYPE:
    UNIFORM_TEMPORAL_SUBSAMPLE = 'uniform_temporal_subsample'
    TIME_STEP_SAMPLE = 'time_step_sample'


class VideoDecoder(PyOperator):
    '''
    VideoDecoder
        Return images with RGB format.
    '''

    def __init__(self, start_time=None, end_time=None, sample_type=None, args=None) -> None:
        super().__init__()
        self._start_time = start_time if start_time is not None else 0
        self._end_time = end_time if end_time is not None else None
        self._end_time_ms = end_time * 1000 if end_time is not None else None
        self._sample_type = sample_type.lower() if sample_type else None
        self._args = args if args is not None else {}

    def decode(self, video_path: str):
        yield from PyAVDecode(video_path, self._start_time).decode()

    def time_step_decode(self, video_path, time_step):
        yield from PyAVDecode(video_path, self._start_time, time_step).time_step_decode()

    def _uniform_temporal_subsample(self, frames, num_samples, total_frames):
        indexs = np.linspace(0, total_frames - 1, num_samples).astype('int')
        cur_index = 0
        count = 0
        for frame in frames:
            if cur_index >= len(indexs):
                return

            while cur_index < len(indexs) and indexs[cur_index] <= count:
                cur_index += 1
                yield frame
            count += 1

        if cur_index < len(indexs):
            yield frame            

    def _filter(self, frames):
        for f in frames:
            if self._end_time_ms and f.timestamp > self._end_time_ms:
                break
            yield f

    def frame_nums(self, video_path):
        with av.open(video_path) as c:
            video = c.streams.video[0]
            start = self._start_time if self._start_time is not None else 0
            duration = c.duration / 1000000
            end = self._end_time if self._end_time and self._end_time <= duration else duration
            return int(round((end - start) * video.average_rate))

    def __call__(self, video_path: str):
        if self._sample_type is None:
            yield from self._filter(self.decode(video_path))
        elif self._sample_type == SAMPLE_TYPE.TIME_STEP_SAMPLE:
            time_step = self._args.get('time_step')
            if time_step is None:
                raise RuntimeError('time_step_sample sample lost args time_step')
            yield from self._filter(self.time_step_decode(video_path, time_step))
        elif self._sample_type == SAMPLE_TYPE.UNIFORM_TEMPORAL_SUBSAMPLE:
            num_samples = self._args.get('num_samples')
            if num_samples is None:
                raise RuntimeError('uniform_temporal_subsample lost args num_samples')
            yield from self._uniform_temporal_subsample(self.decode(video_path), num_samples, self.frame_nums(video_path))
        else:
            raise RuntimeError('Unkown sample type, only supports: [%s|%s]' % (SAMPLE_TYPE.TIME_STEP_SAMPLE, SAMPLE_TYPE.UNIFORM_TEMPORAL_SUBSAMPLE))
