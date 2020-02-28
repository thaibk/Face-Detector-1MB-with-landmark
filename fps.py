# import the necessary packages
import datetime

class FPS:
    def __init__(self, nframes=20):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
        
        self._nframes = nframes
        self._start_n = None
        self.fps_n = 0.0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        if (self._numFrames) % self._nframes == 0:
            self._start_n = datetime.datetime.now()
        
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()

    def get_fps_n(self):
        # run after update
        if (self._numFrames) % self._nframes == 0:
            self.fps_n = self._nframes / ((datetime.datetime.now() - self._start_n).total_seconds())
        return self.fps_n