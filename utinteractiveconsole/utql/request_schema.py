from atom.api import Typed, List, Dict, Str
from .base_schema import (UTQLSchemaEntity, DataflowConfiguration, IDType,
                          AbstractPattern, AbstractNode, AbstractEdge)


class UTQLRequest(UTQLSchemaEntity):
    name = Str()
    patterns = List()


class RequestQueryNode(AbstractNode):
    predicates = Dict()


class RequestOutputNodeType(AbstractNode):
    id = IDType()
    attributes = List()


class QueryEdgeType(AbstractEdge):
    predicates = Dict()


class RequestPattern(AbstractPattern):
    nodes = Dict()

    constraints = List()

    input_edges = List()
    output_edges = List()

    dataflow_config = Typed(DataflowConfiguration)
