from atom.api import Atom, Typed, List, Dict, Enum, Str, Int, Float
import inspect
import numpy as np


class ValueTypeBase(Atom):
    name = Str()
    displayName = Str()
    description = Str()

class EnumValueType(ValueTypeBase):
    value = Typed(Enum)


class IntValueType(ValueTypeBase):
    value = Int()
    default = Int()


class DoubleValueType(ValueTypeBase):
    value = Typed(Float)
    default = Float()


class AttributeDeclarationTypeBase(Atom):
    name = Str()
    displayName = Str()
    description = Str()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        pass

    @classmethod
    def xml_read(cls, node, namespaces, references=None):
        attrs = dict(
            name=node.get('name'),
            displayName=node.get('displayName', ''),
        )

        cls.xml_get_attributes(node, attrs)
        return cls(**attrs)


class StructAttributeDeclarationType(AttributeDeclarationTypeBase):
    value = Typed(AttributeDeclarationTypeBase)
    default = List()


class ListAttributeDeclarationType(AttributeDeclarationTypeBase):
    value = Typed(AttributeDeclarationTypeBase)
    default = List()
    minlen = Int(-1)
    maxlen = Int(-1)


class StringAttributeDeclarationType(AttributeDeclarationTypeBase):
    value = Str()
    default = Str()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['default'] = node.get('default', '')
        attrs['value'] = node.get('value', '')


class ExtendedStringAttributeDeclarationType(AttributeDeclarationTypeBase):
    value = Str()
    default = Str()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['default'] = node.get('default', '')
        attrs['value'] = node.get('value', '')


class HexAttributeDeclarationType(AttributeDeclarationTypeBase):
    value = Int()
    default = Int()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['default'] = int(node.get('default', 0), 16)
        attrs['value'] = int(node.get('value', 0), 16)


class PathAttributeDeclarationType(AttributeDeclarationTypeBase):
    value = Str()
    default = Str()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['default'] = node.get('default', '')
        attrs['value'] = node.get('value', '')


class EnumAttributeDeclarationType(AttributeDeclarationTypeBase):
    value = Str() # XXX Temporary ...
    enum_values = List()
    default = Str()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['default'] = node.get('default', '')
        #attrs['value'] = node.get('value', '')
         # XXX Missing enum values


class IntAttributeDeclarationType(AttributeDeclarationTypeBase):
    value = Int()
    min_value = Float(-np.inf)
    max_value = Float(np.inf)
    default = Int()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['default'] = int(node.get('default', 0))
        attrs['value'] = int(node.get('value', 0))


class DoubleAttributeDeclarationType(AttributeDeclarationTypeBase):
    value = Float()
    min_value = Float(-np.inf)
    max_value = Float(np.inf)
    default = Float()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['default'] = float(node.get('default', 0))
        attrs['value'] = float(node.get('value', 0))


class IntArrayAttributeDeclarationType(AttributeDeclarationTypeBase):
    value = Typed(IntValueType)
    default = List()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['default'] = []
        # XXX TBD


class DoubleArrayAttributeDeclarationType(AttributeDeclarationTypeBase):
    value = Typed(DoubleValueType)
    default = List()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['default'] = []
        # XXX TBD


class AttributeReferenceTypeBase(Atom):
    object = Typed(AttributeDeclarationTypeBase)

    name = Str()
    displayName = Str()
    refersTo = Str()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        pass

    @classmethod
    def xml_read(cls, node, namespaces, references=None):
        if references is None:
            references = {}

        attr_name = node.get('name')
        attr_type = node.get('{http://www.w3.org/2001/XMLSchema-instance}type', '')

        attrs = dict(
            name=attr_name,
            displayName=node.get('displayName', ''),
            refersTo=attr_type,
        )

        obj = references.get((attr_type, attr_name), None)
        if obj is not None:
            attrs['object'] = obj

        cls.xml_get_attributes(node, attrs)
        return cls(**attrs)


class StructAttributeReferenceType(AttributeReferenceTypeBase):
    value = Typed(StructAttributeDeclarationType)


class ListAttributeReferenceType(AttributeReferenceTypeBase):
    value = Typed(ListAttributeDeclarationType)


class StringAttributeReferenceType(AttributeReferenceTypeBase):
    value = Typed(StringAttributeDeclarationType)
    value = Str()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['value'] = node.get('value', '')


class ExtendedStringAttributeReferenceType(AttributeReferenceTypeBase):

    @property
    def value(self):
        if self.object:
            return self.object.default


class HexAttributeReferenceType(AttributeReferenceTypeBase):
    value = Typed(HexAttributeDeclarationType)
    value = Int()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['value'] = int(node.get('value', 0), 16)


class PathAttributeReferenceType(AttributeReferenceTypeBase):
    value = Typed(PathAttributeDeclarationType)
    value = Str()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['value'] = node.get('value', '')


class EnumAttributeReferenceType(AttributeReferenceTypeBase):
    value = Typed(EnumAttributeDeclarationType)
    value = Str()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['value'] = node.get('value', '')
        # XXX ???


class IntAttributeReferenceType(AttributeReferenceTypeBase):
    value = Typed(IntAttributeDeclarationType)
    value = Int()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['value'] = int(node.get('value', 0))


class DoubleAttributeReferenceType(AttributeReferenceTypeBase):
    value = Typed(DoubleAttributeDeclarationType)
    value = Float()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['value'] = float(node.get('value', 0))


class IntArrayAttributeReferenceType(AttributeReferenceTypeBase):
    value = Typed(IntArrayAttributeDeclarationType)
    value = List()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        #attrs['value'] = node.get('value', '')
        # XXX TBD
        pass


class DoubleArrayAttributeReferenceType(AttributeReferenceTypeBase):
    value = Typed(DoubleArrayAttributeDeclarationType)
    value = List()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        #attrs['value'] = node.get('value', '')
        # XXX TBD
        pass


class AttributeTypeBase(Atom):
    name = Str()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        pass

    @classmethod
    def xml_read(cls, node, namespaces, references=None):
        attr_name = node.get('name')

        attrs = dict(
            name=attr_name,
        )
        cls.xml_get_attributes(node, attrs)
        return cls(**attrs)


class PrimitiveAttributeType(AttributeTypeBase):
    value = Str()

    @classmethod
    def xml_get_attributes(cls, node, attrs):
        attrs['value'] = node.get('value', '')


ATTRIBUTE_TYPE_DECLARATION_REGISTRY = {c.__name__: c for c in globals().values() if inspect.isclass(c) and
                                       issubclass(c, AttributeDeclarationTypeBase) and
                                       c is not AttributeDeclarationTypeBase}

ATTRIBUTE_TYPE_REFERENCE_REGISTRY = {c.__name__: c for c in globals().values() if inspect.isclass(c) and
                                     issubclass(c, AttributeReferenceTypeBase) and
                                     c is not AttributeReferenceTypeBase}

ATTRIBUTE_TYPE_REGISTRY = {c.__name__: c for c in globals().values() if inspect.isclass(c) and
                           issubclass(c, AttributeTypeBase) and
                           c is not AttributeTypeBase}
