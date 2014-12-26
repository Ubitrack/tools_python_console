from atom.api import Typed, List, Dict, Str
from .base_schema import (UTQLSchemaEntity, DataflowConfiguration, IDType, IDRef,
                          AbstractPattern, AbstractNode, AbstractEdge)


class UTQLResponse(UTQLSchemaEntity):
    name = Str()
    patterns = List()


class AbstractResponseNode(AbstractNode):
    attributes = List()


class ResponseInputNode(AbstractResponseNode):
    id = IDType()


class ResolvedEdge(AbstractEdge):
    pattern_ref = IDRef()
    edge_ref = IDRef()

    attributes = List()


class ResponseOutputNode(AbstractResponseNode):
    id = IDType()


class ResponsePattern(AbstractPattern):
    id = IDType()

    nodes = Dict()

    input_edges = List()
    output_edges = List()

    dataflow_config = Typed(DataflowConfiguration)
