import os
from lxml import etree

import logging

log = logging.getLogger(__name__)

from .pattern_schema import (PatternTemplate, NodeTemplate, EdgeTemplate, DataflowConfiguration)
from .base_schema import (Correspondence, OnlyBestEdgeMatch, TriggerGroup)
from .attribute_schema import ATTRIBUTE_TYPE_DECLARATION_REGISTRY, ATTRIBUTE_TYPE_REFERENCE_REGISTRY
from .base_parser import (parse_description, parse_attributes, parse_predicates,
                          FileResolver, UTQL_NAMESPACES)


def find_all_patterns(root_path):
    all_xml_files = []
    for root, dirs, files in os.walk(root_path):
        for name in files:
            if os.path.splitext(name)[-1] == '.xml':
                all_xml_files.append(os.path.join(root, name))
    return all_xml_files


def load_global_referencetypes(doc, type_registry, namespaces):
    references = {}
    root = doc.getroot()
    gnad = root.find("utql:GlobalNodeAttributeDeclarations", namespaces=namespaces)
    if gnad is not None:
        reg = references.setdefault('node_attributes', {})
        for attr in gnad.xpath("utql:Attribute", namespaces=namespaces):
            typename = attr.get('{http://www.w3.org/2001/XMLSchema-instance}type')
            fieldname = attr.get('name')
            attr_cls = type_registry.get(typename)
            reg[(typename, fieldname)] = attr_cls.xml_read(attr, namespaces)

    gead = root.find("utql:GlobalEdgeAttributeDeclarations", namespaces=namespaces)
    if gead is not None:
        reg = references.setdefault('edge_attributes', {})
        for attr in gead.xpath("utql:Attribute", namespaces=namespaces):
            typename = attr.get('{http://www.w3.org/2001/XMLSchema-instance}type')
            fieldname = attr.get('name')
            attr_cls = type_registry.get(typename)
            reg[(typename, fieldname)] = attr_cls.xml_read(attr, namespaces)

    gdad = root.find("utql:GlobalDataflowAttributeDeclarations", namespaces=namespaces)
    if gdad is not None:
        reg = references.setdefault('dataflow_attributes', {})
        for attr in gdad.xpath("utql:Attribute", namespaces=namespaces):
            typename = attr.get('{http://www.w3.org/2001/XMLSchema-instance}type')
            fieldname = attr.get('name')
            attr_cls = type_registry.get(typename)
            reg[(typename, fieldname)] = attr_cls.xml_read(attr, namespaces)

    return references


def parse(fname, namespaces=UTQL_NAMESPACES):
    # XXX register namespaces with parser ??
    parser = etree.XMLParser()
    parser.resolvers.add(FileResolver())
    doc = etree.parse(open(fname), parser)
    doc.xinclude()

    root = doc.getroot()
    if not etree.QName(root.tag).localname == 'UTQLPatternTemplates':
        log.error("Invalid root in XML file: %s" % fname)
        return None

    type_registry = {}
    type_registry.update(ATTRIBUTE_TYPE_DECLARATION_REGISTRY)
    type_registry.update(ATTRIBUTE_TYPE_REFERENCE_REGISTRY)

    references = load_global_referencetypes(doc, type_registry, namespaces)

    patterns = {}

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
                attrs = parse_attributes(node, namespaces, type_registry, references.get('node_attributes', {}))
                nodes[node_name] = NodeTemplate(name=node_name,
                                                description=parse_description(node, namespaces),
                                                attributes=attrs,
                                                displayName=node.get("displayName"),
                                                group='output')

            for edge in output.xpath("utql:Edge", namespaces=namespaces):
                attrs = parse_attributes(edge, namespaces, type_registry, references.get('edge_attributes', {}))
                output_edges.append(EdgeTemplate(name=edge.get("name"),
                                                 source=edge.get("source"),
                                                 destination=edge.get("destination"),
                                                 description=parse_description(edge, namespaces),
                                                 displayName=edge.get("displayName"),
                                                 attributes=attrs))

        input = pattern.find("utql:Input", namespaces=namespaces)
        if input is not None:
            for node in input.xpath("utql:Node", namespaces=namespaces):
                node_name = node.get("name")
                attrs = parse_attributes(node, namespaces, type_registry, references.get('node_attributes', {}))
                nodes[node_name] = NodeTemplate(name=node_name,
                                                description=parse_description(node, namespaces),
                                                attributes=attrs,
                                                displayName=node.get("displayName"),
                                                group='input')

            for edge in input.xpath("utql:Edge", namespaces=namespaces):
                attrs = parse_attributes(edge, namespaces, type_registry, references.get('edge_attributes', {}))
                input_edges.append(EdgeTemplate(name=edge.get("name"),
                                                source=edge.get("source"),
                                                destination=edge.get("destination"),
                                                description=parse_description(edge, namespaces),
                                                predicates=parse_predicates(edge, namespaces),
                                                displayName=edge.get("displayName"),
                                                attributes=attrs))

        constraints_element = pattern.find("utql:Constraints", namespaces=namespaces)
        if constraints_element is not None:
            for item in constraints_element.xpath("utql:OnlyBestEdgeMatch", namespaces=namespaces):
                constraints.append(OnlyBestEdgeMatch.xml_read(item, namespaces, references))
            for item in constraints_element.xpath("utql:Correspondence", namespaces=namespaces):
                constraints.append(Correspondence.xml_read(item, namespaces, references))
            for item in constraints_element.xpath("utql:TriggerGroup", namespaces=namespaces):
                constraints.append(TriggerGroup.xml_read(item, namespaces, references))

        dataflowconfig = pattern.find("utql:DataflowConfiguration", namespaces=namespaces)
        if dataflowconfig is not None:
            lib_element = dataflowconfig.find("utql:UbitrackLib", namespaces=namespaces)
            if lib_element is None:
                log.error("Missing ubitrack library name: %s" % pattern_name)
            clsname = lib_element.get('class')
            attrs = parse_attributes(dataflowconfig, namespaces, type_registry, references.get('dataflow_attributes', {}))
            dataflow_configuration = DataflowConfiguration(class_name=clsname, attributes=attrs)

        patterns[pattern_name] = PatternTemplate(name=pattern_name, nodes=nodes,
                                                 displayName=pattern.get('displayName', ''),
                                                 input_edges=input_edges, output_edges=output_edges,
                                                 dataflow_config=dataflow_configuration, constraints=constraints,
                                                 type_registry=type_registry, references=references)

    return patterns