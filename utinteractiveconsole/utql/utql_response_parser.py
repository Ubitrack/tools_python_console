__author__ = 'jack'

from lxml import etree

import logging

log = logging.getLogger(__name__)

from .response_schema import (UTQLResponse, ResponsePattern, ResponseOutputNode, ResponseInputNode, ResolvedEdge)
from .base_schema import (OutputEdge, DataflowConfiguration)
from .attribute_schema import ATTRIBUTE_TYPE_REGISTRY
from .base_parser import (parse_description, parse_attributes, parse_gui_status,
                          FileResolver, UTQL_NAMESPACES)


def parse(fname, namespaces=UTQL_NAMESPACES):
    # XXX register namespaces with parser ??
    parser = etree.XMLParser()
    parser.resolvers.add(FileResolver())
    doc = etree.parse(open(fname), parser)
    doc.xinclude()

    root = doc.getroot()
    if not etree.QName(root.tag).localname == 'UTQLResponse':
        log.error("Invalid root in XML file: %s" % fname)
        return None

    type_registry = {}
    type_registry.update(ATTRIBUTE_TYPE_REGISTRY)

    response_name = root.get('name')

    patterns = []

    for pattern in root.xpath("utql:Pattern", namespaces=namespaces):
        pattern_name = pattern.get('name')
        nodes = {}
        input_edges = []
        output_edges = []
        dataflow_configuration = None

        output = pattern.find("utql:Output", namespaces=namespaces)
        if output is not None:
            for node in output.xpath("utql:Node", namespaces=namespaces):
                node_name = node.get("name")
                gui_status = parse_gui_status(node, namespaces)
                nodes[node_name] = ResponseOutputNode(name=node.get("name"),
                                                          displayName=node.get("displayName", ""),
                                                          description=parse_description(node, namespaces),
                                                          id=node.get("id", ""),
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
                nodes[node_name] = ResponseInputNode(name=node_name,
                                                         displayName=node.get("displayName", ""),
                                                         description=parse_description(node, namespaces),
                                                         id=node.get("id"),
                                                         gui_pos=gui_status.get('gui_pos'),
                                                         attributes=parse_attributes(node, namespaces, type_registry),
                                                         group='input')

            for edge in input.xpath("utql:Edge", namespaces=namespaces):
                gui_status = parse_gui_status(edge, namespaces)
                input_edges.append(ResolvedEdge(name=edge.get("name"),
                                                    displayName=edge.get("displayName", ""),
                                                    description=parse_description(edge, namespaces),
                                                    source=edge.get("source"),
                                                    destination=edge.get("destination"),
                                                    pattern_ref=edge.get("pattern-ref"),
                                                    edge_ref=edge.get("edge-ref"),
                                                    gui_label_pos=gui_status.get('gui_label_pos'),
                                                    gui_landmark=gui_status.get('gui_landmark'),
                                                    attributes=parse_attributes(edge, namespaces, type_registry)))

        dataflowconfig = pattern.find("utql:DataflowConfiguration", namespaces=namespaces)
        if dataflowconfig is not None:
            lib_element = dataflowconfig.find("utql:UbitrackLib", namespaces=namespaces)
            if lib_element is None:
                log.error("Missing ubitrack library name: %s" % pattern_name)
            clsname = lib_element.get('class')
            attrs = parse_attributes(dataflowconfig, namespaces, type_registry)
            dataflow_configuration = DataflowConfiguration(class_name=clsname, attributes=attrs)

        patterns.append(ResponsePattern(name=pattern_name,
                                        id=pattern.get("id"),
                                        nodes=nodes,
                                        displayName=pattern.get('displayName', ''),
                                        input_edges=input_edges,
                                        output_edges=output_edges,
                                        dataflow_config=dataflow_configuration))

    return UTQLResponse(name=response_name, patterns=patterns)