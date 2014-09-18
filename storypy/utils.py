from collections import defaultdict
from itertools import combinations

class Encoder(defaultdict):

    def __init__(self, items=[]):
        defaultdict.__init__(self, None)
        self.default_factory = self.__len__
        self.update(items)

    def add(self, item):
        self[item]

    def update(self, items):
        for item in items:
            self.add(item)

def mean(lst):
	return sum(lst) / float(len(lst))
