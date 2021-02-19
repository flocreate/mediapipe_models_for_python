from datetime import datetime, timedelta
from collections import OrderedDict, deque
# #########################################


class Chronometer:
    def __init__(self, history_len=None):
        self.duration   = timedelta()
        self.measures   = deque(maxlen=history_len)
        self._start     = None
        self.childs     = OrderedDict()
    # #########################################

    def __repr__(self):
        return "<Chronometer %s>" % str(self.duration)
    # #########################################

    def clear(self):
        self.duration = timedelta()
        self.measures.clear()
        for child in self.childs.values():
            child.clear()
    # #########################################

    def start(self):
        self._start = datetime.now()
        return self
    # #########################################

    def stop(self):
        duration = datetime.now() - self._start
        self.measures.append(duration)
        self.duration = sum(self.measures, timedelta()) / len(self.measures)
        return self
    # #########################################


    def __enter__(self):
        return self.start()
    # #########################################

    def __exit__(self, *_):
        self.stop()
    # #########################################

    def __getitem__(self, key):
        if key in self.childs:
            child = self.childs[key]
        else:            
            child = Chronometer(self.measures.maxlen)
            self.childs[key] = child
        return child
    # #########################################

    def as_dict(self):
        if len(self.childs):
            return OrderedDict([
                ("total", self.duration.total_seconds()),
                * [
                    (child_name, child.as_dict())
                    for child_name, child
                    in self.childs.items()
                ]
            ])
        else:
            return self.duration.total_seconds()
    # #########################################
