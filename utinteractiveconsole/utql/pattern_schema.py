from atom.api import Typed, List, Dict, Enum, Str
from .base_schema import UTQLSchemaEntity, DataflowConfiguration


class NodeTemplate(UTQLSchemaEntity):

    name = Str()
    displayName = Str()

    group = Enum('input', 'output')

    description = Str()

    attributes = List()


class EdgeTemplate(UTQLSchemaEntity):

    name = Str()
    displayName = Str()

    description = Str()

    source = Str()
    destination = Str()

    predicates = Dict()

    attributes = List()


class PatternTemplate(UTQLSchemaEntity):

    name = Str()
    displayName = Str()
    description = Str()

    nodes = Dict()

    constraints = List()

    input_edges = List()
    output_edges = List()

    type_registry = Dict()
    references = Dict()

    dataflow_config = Typed(DataflowConfiguration)

