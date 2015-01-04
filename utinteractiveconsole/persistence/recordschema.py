__author__ = 'jack'

from atom.api import Atom, List, Str, Int, Float, IntEnum, Bool, Typed, Coerced


class DataType(IntEnum):
    # ubitrack datatypes
    distance = 0x00
    position2d = 0x01
    position3d = 0x02
    quat = 0x03
    pose = 0x04
    mat33 = 0x05
    mat34 = 0x06
    mat44 = 0x07


class SelectorType(IntEnum):
    matching = 0x00
    nearest = 0x01
    interpolate = 0x02
    index = 0x03


class Field(Atom):
    name = Str()
    datatype = Coerced(DataType.Flags)
    is_array = Bool()

    latency = Float(0.0)
    selector = Coerced(SelectorType.Flags)
    interval = Float(1.0)

    is_computed = Bool(False)
    is_reference = Bool(False)


class RequiredField(Atom):
    name = Str()
    datatype = Coerced(DataType.Flags)
    is_array = Bool(False)


class RecordSchema(Atom):
    fields = List()

    def as_dataframe(self):
        from pandas import DataFrame
        name = []
        datatype = []
        is_array = []
        latency = []
        selector = []
        is_computed = []
        is_reference = []

        for field in self.fields:
            name.append(field.name)
            datatype.append(field.datatype)
            is_array.append(field.is_array)
            latency.append(field.latency)
            selector.append(field.selector)
            is_computed.append(field.is_computed)
            is_reference.append(field.is_reference)

        return DataFrame(dict(name=name,
                              datatype=datatype,
                              is_array=is_array,
                              latency=latency,
                              selector=selector,
                              is_computed=is_computed,
                              is_reference=is_reference,
                              ),
                         columns=['name', 'datatype', 'is_array', 'latency', 'selector', 'is_computed', 'is_reference'],
                         index=range(len(name)),
                         )
