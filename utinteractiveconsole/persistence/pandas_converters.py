__author__ = 'jack'

from atom.api import Atom, Bool, Int

import pandas as pd
import numpy as np

from ubitrack.core import math
from .recordschema import DataType

# XXX use table format by default
pd.set_option('io.hdf.default_format', 'table')



class PandasTypeConverter(Atom):
    is_array = Bool(False)
    element_count = Int(1)

    def to_dataframe(self, value):
        raise NotImplemented()

    def from_dataframe(self, df):
        raise NotImplemented()


class DistanceTypeConverter(PandasTypeConverter):

    def to_dataframe(self, value):
        if self.element_count != 1:
            raise TypeError("Multivalue records for distance are not supported yet.")
        if not self.is_array:
            value = [value, ]
        return pd.DataFrame(np.array(value, dtype=np.double),
                            columns=['value'])

    def from_dataframe(self, df):
        if self.is_array:
            return df.icol(0).tolist()
        else:
            return df.iloc[0][0]


class Position2dTypeConverter(PandasTypeConverter):

    def to_dataframe(self, value):
        if self.element_count != 1:
            raise TypeError("Multivalue records for position2d are not supported yet.")
        if not self.is_array:
            value = [value, ]
        return pd.DataFrame(np.asarray(value), columns=['x', 'y'])

    def from_dataframe(self, df):
        if self.is_array:
            return list(np.asarray(df))
        else:
            return np.asarray(df.iloc[0])


class Position3dTypeConverter(PandasTypeConverter):

    def to_dataframe(self, value):
        if not self.is_array:
            value = [value, ]

        if self.element_count == 1:
            return pd.DataFrame(np.asarray(value), columns=['x', 'y', 'z'])
        else:
            columns = []
            for i in range(self.element_count):
                columns.append('x_%02d' % i)
                columns.append('y_%02d' % i)
                columns.append('z_%02d' % i)
            return pd.DataFrame(np.asarray([np.hstack(v) for v in value]), columns=columns)

    def from_dataframe(self, df):
        if self.element_count == 1:
            if self.is_array:
                return list(np.asarray(df))
            else:
                return np.asarray(df.iloc[0])
        else:
            groups = []
            for i in range(self.element_count):
                groups.append(['x_%02d' % i, 'y_%02d' % i, 'z_%02d' % i])

            def unpack(v):
                r = []
                for group in groups:
                    r.append(np.asarray([v[c] for c in group]))
                return r

            if self.is_array:
                return [unpack(record) for idx, record in df.iterrows()]
            else:
                return unpack(np.asarray(df.iloc[0]))



class QuaternionTypeConverter(PandasTypeConverter):

    def to_dataframe(self, value):
        if self.element_count != 1:
            raise TypeError("Multivalue records for quaternion are not supported yet.")
        data = []
        if not self.is_array:
            value = [value, ]
        for v in value:
            data.append(tuple(v.toVector()))
        return pd.DataFrame(np.array(data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('w', 'f4')]))

    def from_dataframe(self, df):
        if self.is_array:
            return [math.Quaternion.fromVector(list(v)) for v in np.asarray(df)]
        else:
            return math.Quaternion.fromVector(list(df.iloc[0]))


class PoseTypeConverter(PandasTypeConverter):

    def to_dataframe(self, value):
        if self.element_count != 1:
            raise TypeError("Multivalue records for pose are not supported yet.")
        data = []
        if not self.is_array:
            value = [value, ]
        for v in value:
            data.append(tuple(v.toVector()))
        return pd.DataFrame(np.array(data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                  ('rx', 'f4'), ('ry', 'f4'), ('rz', 'f4'), ('rw', 'f4')]))

    def from_dataframe(self, df):
        if self.is_array:
            return [math.Pose.fromVector(list(v)) for v in np.asarray(df)]
        else:
            return math.Pose.fromVector(list(df.iloc[0]))


class MatrixTypeConverter(PandasTypeConverter):

    def to_dataframe(self, value):
        if self.element_count != 1:
            raise TypeError("Multivalue records for matrix are not supported yet.")
        if not self.is_array:
            value = [value, ]
        return pd.Panel(np.asarray(value))

    def from_dataframe(self, df):
        if self.is_array:
            return list(np.asarray(df))
        else:
            return np.asarray(df.iloc[0])


type_converters = {
    DataType.distance: DistanceTypeConverter,
    DataType.position2d: Position2dTypeConverter,
    DataType.position3d: Position3dTypeConverter,
    DataType.quat: QuaternionTypeConverter,
    DataType.pose: PoseTypeConverter,
    DataType.mat33: MatrixTypeConverter,
    DataType.mat34: MatrixTypeConverter,
    DataType.mat44: MatrixTypeConverter,
}


def get_typeconverter(datatype, is_array=False, element_count=1):
    return type_converters[datatype](is_array=is_array, element_count=element_count)


def guess_type(value):
    if isinstance(value, float):
        return DataType.distance
    elif isinstance(value, (np.ndarray,)):
        if value.shape == (2,):
            return DataType.position2d
        elif value.shape == (3,):
            return DataType.position3d
        elif value.shape == (3, 3):
            return DataType.mat33
        elif value.shape == (3, 4):
            return DataType.mat34
        elif value.shape == (4, 4):
            return DataType.mat44
        else:
            raise TypeError("Unsupported array-type: %s" % (value.shape, ))
    elif isinstance(value, math.Pose):
        return DataType.pose
    elif isinstance(value, math.Quaternion):
        return DataType.quat
    else:
        raise TypeError("Unsupported type: %s" % type(value))


def store_data(store, path, value, datatype=None, is_array=False, element_count=1):
    if datatype is not None:
        converter = get_typeconverter(datatype, is_array=is_array, element_count=element_count)
        data = converter.to_dataframe(value)
    else:
        if not isinstance(value, (pd.DataFrame, pd.Panel, pd.Series)):
            data = pd.DataFrame(value)
        else:
            data = value
    store.put(path, data)
    st = store.get_storer(path)
    st.attrs.utic_datatype = datatype
    st.attrs.utic_is_array = is_array
    st.attrs.utic_element_count = element_count


def load_data(store, path):
    st = store.get_storer(path)
    datatype = st.attrs.utic_datatype
    is_array = st.attrs.utic_is_array
    element_count = st.attrs.utic_element_count

    data = store.get(path)

    if datatype is not None:
        converter = get_typeconverter(datatype, is_array=is_array, element_count=element_count)
        data = converter.from_dataframe(data)

    return data
