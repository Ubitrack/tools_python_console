__author__ = 'jack'

from lxml import etree

import logging

log = logging.getLogger(__name__)

from .request_schema import (UTQLRequest, RequestPattern, RequestQueryNode, RequestOutputNodeType, QueryEdgeType)
from .base_schema import (OutputEdge, Correspondence, OnlyBestEdgeMatch, TriggerGroup, DataflowConfiguration)
from .attribute_schema import ATTRIBUTE_TYPE_REGISTRY
from .base_parser import (parse_description, parse_attributes, parse_predicates, parse_gui_status,
                          FileResolver, UTQL_NAMESPACES)


def parse(fname, namespaces=UTQL_NAMESPACES):
    # XXX register namespaces with parser ??
    parser = etree.XMLParser()
    parser.resolvers.add(FileResolver())
    doc = etree.parse(open(fname), parser)
    doc.xinclude()

    root = doc.getroot()
    if not etree.QName(root.tag).localname == 'UTQLRequest':
        log.error("Invalid root in XML file: %s" % fname)
        return None

    type_registry = {}
    type_registry.update(ATTRIBUTE_TYPE_REGISTRY)

    request_name = root.get('name')

    patterns = []

    for pattern in root.xpath("utql:Pattern", namespaces=namespaces):
        pattern_name = pattern.get('name')
        nodes = {}
        input_edges = []
        output_edges = []
        dataflow_configuration = None
        constraints = []

        output = pattern.find("utql:Output", namespaces=namespaces)
        if output is not None:
            for node in output.xpath("utql:Node", namespaces=namespaces):
                node_name = node.get("name")
                gui_status = parse_gui_status(node, namespaces)
                nodes[node_name] = RequestOutputNodeType(id=node.get("id"),
                                                         name=node.get("name"),
                                                         displayName=node.get("displayName", ""),
                                                         description=parse_description(node, namespaces),
                                                         attributes=parse_attributes(node, namespaces, type_registry),
                                                         gui_pos=gui_status.get('gui_pos'),
                                                         group='output')

            for edge in output.xpath("utql:Edge", namespaces=namespaces):
                gui_status = parse_gui_status(edge, namespaces)
                output_edges.append(OutputEdge(name=edge.get("name"),
                                                   displayName=edge.get("displayName", ""),
                                                   description=parse_description(edge, namespaces),
                                                   source=edge.get("source"),
                                                   destination=edge.get("destination"),
                                                   gui_label_pos=gui_status.get('gui_label_pos'),
                                                   gui_landmark=gui_status.get('gui_landmark'),
                                                   attributes=parse_attributes(edge, namespaces, type_registry)))

        input = pattern.find("utql:Input", namespaces=namespaces)
        if input is not None:
            for node in input.xpath("utql:Node", namespaces=namespaces):
                node_name = node.get("name")
                gui_status = parse_gui_status(node, namespaces)
                nodes[node_name] = RequestQueryNode(name=node_name,
                                                    displayName=node.get("displayName", ""),
                                                    description=parse_description(node, namespaces),
                                                    gui_pos=gui_status.get('gui_pos'),
                                                    group='input')

            for edge in input.xpath("utql:Edge", namespaces=namespaces):
                gui_status = parse_gui_status(edge, namespaces)
                input_edges.append(QueryEdgeType(name=edge.get("name"),
                                                 displayName=edge.get("displayName", ""),
                                                 description=parse_description(edge, namespaces),
                                                 source=edge.get("source"),
                                                 destination=edge.get("destination"),
                                                 predicates=parse_predicates(edge, namespaces),
                                                 gui_label_pos=gui_status.get('gui_label_pos'),
                                                 gui_landmark=gui_status.get('gui_landmark')))

        constraints_element = pattern.find("utql:Constraints", namespaces=namespaces)
        if constraints_element is not None:
            for item in constraints_element.xpath("utql:OnlyBestEdgeMatch", namespaces=namespaces):
                constraints.append(OnlyBestEdgeMatch.xml_read(item, namespaces))
            for item in constraints_element.xpath("utql:Correspondence", namespaces=namespaces):
                constraints.append(Correspondence.xml_read(item, namespaces))
            for item in constraints_element.xpath("utql:TriggerGroup", namespaces=namespaces):
                constraints.append(TriggerGroup.xml_read(item, namespaces))

        dataflowconfig = pattern.find("utql:DataflowConfiguration", namespaces=namespaces)
        if dataflowconfig is not None:
            lib_element = dataflowconfig.find("utql:UbitrackLib", namespaces=namespaces)
            if lib_element is None:
                log.error("Missing ubitrack library name: %s" % pattern_name)
            clsname = lib_element.get('class')
            attrs = parse_attributes(dataflowconfig, namespaces, type_registry)
            dataflow_configuration = DataflowConfiguration(class_name=clsname, attributes=attrs)

        patterns.append(RequestPattern(name=pattern_name,
                                       nodes=nodes,
                                       displayName=pattern.get('displayName', ''),
                                       input_edges=input_edges,
                                       output_edges=output_edges,
                                       dataflow_config=dataflow_configuration,
                                       constraints=constraints))

    return UTQLRequest(name=request_name, patterns=patterns)