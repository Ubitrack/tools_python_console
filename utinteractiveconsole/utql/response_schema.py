from atom.api import Typed, List, Dict, Str
from .base_schema import (UTQLSchemaEntity, DataflowConfiguration, IDType, IDRef,
                          AbstractPattern, AbstractNode, AbstractEdge)


class UTQLResponse(UTQLSchemaEntity):
    name = Str()
    patterns = List()


class AbstractResponseNode(AbstractNode):
    attributes = List()


class ResponseInputNodeType(AbstractResponseNode):
    id = IDType()


class ResolvedEdgeType(AbstractEdge):
    pattern_ref = IDRef()
    edge_ref = IDRef()

    attributes = List()


class ResponseOutputNodeType(AbstractResponseNode):
    id = IDType()


class ResponsePattern(AbstractPattern):
    id = IDType()

    nodes = Dict()

    input_edges = List()
    output_edges = List()

    dataflow_config = Typed(DataflowConfiguration)
