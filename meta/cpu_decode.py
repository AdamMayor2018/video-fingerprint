import math
import logging
import av

from towhee.types.video_frame import VideoFrame


logger = logging.getLogger()


class PyAVDecode:
    def __init__(self, video_path, start_time=None, time_step=None) -> None:
        self._container = av.open(video_path)
        self._stream = self._container.streams.video[0]
        self._start_time = start_time if start_time is not None else 0
        self._time_step = time_step

    def close(self):
        self._container.close()

    def time_step_decode(self):
        ts = self._start_time
        is_end = False
        while not is_end:
            is_end = True
            offset = int(math.floor(ts / self._stream.time_base))
            self._container.seek(offset, stream=self._stream)
            for f in self._container.decode(self._stream):
                if f.time < ts:
                    continue
                yield self.av_frame_to_video_frame(f)
                is_end = False
                break
            ts += self._time_step

    def av_frame_to_video_frame(self, frame):
        timestamp = int(round(frame.time * 1000))
        ndarray = frame.to_ndarray(format='rgb24')
        img = VideoFrame(ndarray, 'RGB', timestamp, frame.key_frame)
        return img

    def decode(self):
        if self._start_time > 0:
            offset = int(math.floor(self._start_time / self._stream.time_base))
            self._container.seek(offset, any_frame=False, backward=True, stream=self._stream)

        for frame in self._container.decode(self._stream):
            if frame.time < self._start_time:
                continue
            yield self.av_frame_to_video_frame(frame)
