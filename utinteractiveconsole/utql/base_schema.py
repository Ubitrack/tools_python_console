from atom.api import Atom, Value, List, Str, Enum, Int


UTQL_DATATYPES = ["6D", "2DPosition", "3DPosition", "3DRotation", "Distance", "Scalar", "4DVector", "Image", "3x3Matrix", "4x4Matrix", "3x4Matrix", "DistanceList", "3DPositionList", "2DPositionList", "PoseList", "Button", "RotationVelocity", "6DError", "3DPositionError", ]


class UTQLSchemaEntity(Atom):
    pass


class IDType(Str):
    pass


class IDRef(Str):
    pass


class AbstractPattern(UTQLSchemaEntity):
    name = Str()
    displayName = Str()
    description = Str()


class AbstractNode(UTQLSchemaEntity):
    name = Str()
    displayName = Str()
    description = Str()

    gui_pos = Value()
    group = Enum('input', 'output')


class AbstractEdge(UTQLSchemaEntity):
    name = IDType()
    displayName = Str()
    description = Str()

    source = IDType()
    destination = IDType()

    gui_label_pos = Value()
    gui_landmark = Value()


class OutputEdge(AbstractEdge):
    attributes = List()


class ConstraintBase(UTQLSchemaEntity):

    @classmethod
    def xml_get_attributes(cls, node, attrs, namespaces):
        pass

    @classmethod
    def xml_read(cls, node, namespaces, references=None):
        attrs = {}
        cls.xml_get_attributes(node, attrs, namespaces)
        return cls(**attrs)


class OnlyBestEdgeMatch(ConstraintBase):
    pass


class Correspondence(ConstraintBase):
    name = Str()
    minMultiplicity = Int()
    maxMultiplicity = Int()
    stepSize = Int()

    edges = List()
    nodes = List()

    @classmethod
    def xml_get_attributes(cls, node, attrs, namespaces):
        attrs['name'] = node.get('name')
        attrs['minMultiplicity'] = int(node.get('minMultiplicity', -1))
        attrs['maxMultiplicity'] = int(node.get('maxMultiplicity', -1))
        attrs['stepSize'] = int(node.get('stepSize', 1))

        edges = []
        for edge in node.xpath("utql:Edge", namespaces=namespaces):
            edges.append(('edge-ref', edge.get('edge-ref')))

        nodes = []
        for nodes in node.xpath("utql:Node", namespaces=namespaces):
            nodes.append(('nodes-ref', nodes.get('nodes-ref')))

        attrs['edges'] = edges
        attrs['nodes'] = nodes


class TriggerGroup(ConstraintBase):

    edges = List()

    @classmethod
    def xml_get_attributes(cls, node, attrs, namespaces):
        edges = []
        for edge in node.xpath("utql:Edge", namespaces=namespaces):
            edges.append(('edge-ref', edge.get('edge-ref')))

        attrs['edges'] = edges


class DataflowConfiguration(UTQLSchemaEntity):
    class_name = Str()

    attributes = List()

