import time

class Timer(object):
    def __init__(self):
        pass

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print('\n\nTempo di esecuzione (ms): %s' % (time.time() - self.tstart))