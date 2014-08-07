__author__ = 'jack'

import numpy as np
from numpy.linalg import norm
import sys
import logging

from ubitrack.core import math

log = logging.getLogger(__name__)

class BaseStreamFilter(object):

    def process(self, stream):
        return stream



class StaticPointDistanceStreamFilter(BaseStreamFilter):

    def __init__(self, fieldname, point, min_distance=0, max_distance=sys.maxint):
        log.info("StaticPointDistanceStreamFilter point=%s, min_distance=%s, max_distance=%s" %
                (point, min_distance, max_distance))
        self.fieldname = fieldname
        self.point = point
        self.min_distance = min_distance
        self.max_distance = max_distance

    def check_item(self, item):
        if isinstance(item, math.Pose):
            item = item.translation()

        d = norm(self.point - item)

        if d >= self.min_distance and d <= self.max_distance:
            return True
        return False


    def process(self, stream):
        result = []
        selector = self.check_item
        fieldname = self.fieldname

        for record in stream:
            item = getattr(record, fieldname)
            if selector(item):
                result.append(record)
        return result



class RelativePointDistanceStreamFilter(BaseStreamFilter):

    def __init__(self, fieldname, min_distance=0, max_distance=sys.maxint):
        log.info("RelativePointDistanceStreamFilter min_distance=%s, max_distance=%s" %
                (min_distance, max_distance))
        self.fieldname = fieldname
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.last_position = None

    def check_item(self, item):
        if isinstance(item, math.Pose):
            item = item.translation()

        if self.last_position is None:
            self.last_position = item
            return False

        d = norm(self.last_position - item)

        if d >= self.min_distance and d <= self.max_distance:
            self.last_position = item
            return True
        return False


    def process(self, stream):
        result = []
        selector = self.check_item
        fieldname = self.fieldname

        for record in stream:
            item = getattr(record, fieldname)
            if selector(item):
                result.append(record)
        return result


class TwoPointDistanceStreamFilter(BaseStreamFilter):

    def __init__(self, fieldname1, fieldname2, min_distance=0, max_distance=sys.maxint):
        log.info("TwoPointDistanceStreamFilter min_distance=%s, max_distance=%s" %
                (min_distance, max_distance))
        self.fieldname1 = fieldname1
        self.fieldname2 = fieldname2
        self.min_distance = min_distance
        self.max_distance = max_distance

    def check_item(self, item1, item2):
        if isinstance(item1, math.Pose):
            item1 = item1.translation()
        if isinstance(item2, math.Pose):
            item2 = item2.translation()

        d = norm(item1 - item2)

        if d >= self.min_distance and d <= self.max_distance:
            return True
        return False


    def process(self, stream):
        result = []
        selector = self.check_item
        fieldname1 = self.fieldname1
        fieldname2 = self.fieldname2

        for record in stream:
            item1 = getattr(record, fieldname1)
            item2 = getattr(record, fieldname2)
            if selector(item1, item2):
                result.append(record)
        return result



class RelativeOrienationDistanceStreamFilter(BaseStreamFilter):

    def __init__(self, fieldname, min_distance=0, max_distance=sys.maxint):
        log.info("RelativeOrienationDistanceStreamFilter min_distance=%s, max_distance=%s" %
                (min_distance, max_distance))
        self.fieldname = fieldname
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.last_rotation = None

    def check_item(self, item):
        if isinstance(item, math.Pose):
            item = item.rotation()

        if self.last_rotation is None:
            self.last_rotation = item
            return False

        d = abs(math.Quaternion(self.last_rotation.inverted() * item).angle())

        if d >= self.min_distance and d <= self.max_distance:
            self.last_rotation = item
            return True
        return False


    def process(self, stream):
        result = []
        selector = self.check_item
        fieldname = self.fieldname

        for record in stream:
            item = getattr(record, fieldname)
            if selector(item):
                result.append(record)
        return result
