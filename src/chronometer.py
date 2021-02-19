from datetime import datetime, timedelta
from collections import OrderedDict
# #########################################


class Chronometer:
    def __init__(self):
        self.duration   = timedelta()
        self._start     = None
        self.childs     = OrderedDict()
    # ####################################

    def __repr__(self):
        return "<%s %s>" % (type(self), str(self.duration))
    # ####################################

    def start(self):
        self.duration = timedelta()
        self._start = datetime.now()
        return self
    # ####################################

    def __enter__(self):
        return self.start()
    # ####################################

    def stop(self):
        self.duration = datetime.now() - self._start
        return self
    # ####################################

    def __exit__(self, *_):
        self.stop()
    # ####################################

    def __getitem__(self, key):
        if key in self.childs:
            child = self.childs[key]
        else:            
            child = Chronometer()
            self.childs[key] = child
        return child
    # ####################################

    def as_dict(self):
        if len(self.childs):
            return {
                "total": self.duration.total_seconds(),
                ** {
                    child_name: child.as_dict()
                    for child_name, child
                    in self.childs.items()
                }
            }
        else:
            return self.duration.total_seconds()
    # ####################################

    