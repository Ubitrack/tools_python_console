__author__ = 'jack'

import numpy as np
from numpy.linalg import norm
import sys
import logging

from scipy import cluster
from scipy import spatial

from ubitrack.core import math

log = logging.getLogger(__name__)

class BaseStreamFilter(object):

    def process(self, stream):
        for record in stream:
            yield record




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
        selector = self.check_item
        fieldname = self.fieldname

        for record in stream:
            item = getattr(record, fieldname)
            if selector(item):
                yield record


class StaticLineDistanceStreamFilter(BaseStreamFilter):

    def __init__(self, fieldname, point1, point2, min_distance=0, max_distance=sys.maxint):
        log.info("StaticLineDistanceStreamFilter point1=%s, point2=%s, min_distance=%s, max_distance=%s" %
                (point1, point2, min_distance, max_distance))
        self.fieldname = fieldname
        self.point1 = point1
        self.point2 = point2
        self.min_distance = min_distance
        self.max_distance = max_distance

    def check_item(self, item):
        if isinstance(item, math.Pose):
            item = item.translation()

        # closed form solution for point-to-line distance
        # http://stackoverflow.com/questions/19341904/shortest-distance-between-point-and-line-point-w-direction-vector-in-3d-using
        d = norm(np.cross(self.point2-self.point1, self.point1-item))/norm(self.point2-self.point1)

        if d >= self.min_distance and d <= self.max_distance:
            return True
        return False


    def process(self, stream):
        selector = self.check_item
        fieldname = self.fieldname

        for record in stream:
            item = getattr(record, fieldname)
            if selector(item):
                yield record


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
        selector = self.check_item
        fieldname = self.fieldname

        for record in stream:
            item = getattr(record, fieldname)
            if selector(item):
                yield record


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
        selector = self.check_item
        fieldname1 = self.fieldname1
        fieldname2 = self.fieldname2

        for record in stream:
            item1 = getattr(record, fieldname1)
            item2 = getattr(record, fieldname2)
            if selector(item1, item2):
                yield record



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
        selector = self.check_item
        fieldname = self.fieldname

        for record in stream:
            item = getattr(record, fieldname)
            if selector(item):
                yield record


class NClustersPositionStreamFilter(BaseStreamFilter):

    def __init__(self, fieldname, n_clusters):
        log.info("NClustersPositionStreamFilter n=%s" % n_clusters)
        self.fieldname = fieldname
        self.n_clusters = n_clusters

    def process(self, stream):
        fieldname = self.fieldname

        # needs preloading data
        datastream = list(stream)

        timestamps = []
        measurements = []

        for record in datastream:
            timestamps.append(record.timestamp)
            m = getattr(record, fieldname)
            if isinstance(m, math.Pose):
                m = m.translation()
            measurements.append(m)

        measurements = np.asarray(measurements)
        timestamps = np.asarray(timestamps)

        centroids, _ = cluster.vq.kmeans2(measurements, self.n_clusters, minit='points')
        clusters, _ = cluster.vq.vq(measurements, centroids)
        kdt = spatial.cKDTree(measurements)

        selected_timestamps = []
        for v in centroids:
            dist, idx = kdt.query(v)
            selected_timestamps.append(timestamps[idx])

        for record in datastream:
            if record.timestamp in selected_timestamps:
                yield record


class ExcludeTimestampsStreamFilter(BaseStreamFilter):

    def __init__(self, timestamps):
        log.info("ExcludeTimestampsStreamFilter l=%d" % len(timestamps))
        self.timestamps = timestamps

    def process(self, stream):
        excluded_timestamps = self.timestamps

        for record in stream:
            if record.timestamp not in excluded_timestamps:
                yield record