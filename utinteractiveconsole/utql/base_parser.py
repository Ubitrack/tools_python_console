import os
import urlparse
from lxml import etree


UTQL_NAMESPACES = {'utql': 'http://ar.in.tum.de/ubitrack/utql',
                   'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
                   'xi': 'http://www.w3.org/2001/XInclude'}


class FileResolver(etree.Resolver):

    def resolve(self, url, id, context):
        u = urlparse.urlparse(url)
        if not os.path.isfile(u.path):
            filename = os.path.basename(u.path)
            dirname = os.path.dirname(u.path)

            print "Resolving invalid URL '%s' '%s'" % (dirname, filename)

            raise NotImplementedError()
        return super(FileResolver, self).resolve(url, id, context)


def find_all_patterns(root_path):
    all_xml_files = []
    for root, dirs, files in os.walk(root_path):
        for name in files:
            if os.path.splitext(name)[-1] == '.xml':
                all_xml_files.append(os.path.join(root, name))
    return all_xml_files


def parse_description(node, namespaces):
    description = []

    for e in node.xpath("utql:Description", namespaces=namespaces):
        txt = e.xpath("string()", namespaces=namespaces)
        if txt:
            description.append(txt)

    return "\n".join(description)


def parse_gui_status(node, namespaces):
    status = {}

    gui_pos_element = node.find("utql:GuiPos", namespaces=namespaces)
    if gui_pos_element is not None:
        status['gui_pos'] = (int(gui_pos_element.get('x', 0)), int(gui_pos_element.get('y', 0)))

    gui_landmark_element = node.find("utql:GuiLandmark", namespaces=namespaces)
    if gui_landmark_element is not None:
        status['gui_landmark'] = (int(gui_landmark_element.get('x', 0)), int(gui_landmark_element.get('y', 0)))

    gui_label_pos_element = node.find("utql:GuiLabelPos", namespaces=namespaces)
    if gui_label_pos_element is not None:
        status['gui_label_pos'] = (int(gui_label_pos_element.get('x', 0)), int(gui_label_pos_element.get('y', 0)))

    return status


def parse_predicates(node, namespaces):
    predicates = {}

    for e in node.xpath("utql:Predicate", namespaces=namespaces):
        txt = e.xpath("string()", namespaces=namespaces)
        if txt:
            for item in txt.split("&&"):
                k, v = item.split("==")
                predicates[k] = v.replace("'", "").replace('"', "").strip()

    return predicates


def parse_attributes(node, namespaces, type_registry, references=None):
    attrs = []

    # XXX Predicate vs. Attribute = not really clear why ??
    for e in node.xpath("utql:Attribute", namespaces=namespaces):

        # XXX shouldn't use schema url directly maybe .. ???
        tname = e.get('{http://www.w3.org/2001/XMLSchema-instance}type')
        if tname.startswith("utql:"):
            tname = tname.replace("utql:", "")

        attr_cls = type_registry.get(tname)
        if attr_cls is None:
            raise TypeError("Missing AttributeType: %s" % tname)
        attrs.append(attr_cls.xml_read(e, namespaces, references=references))

    return attrs


def attributes_as_dict(attrs):
    result = {}
    for item in attrs:
        result[item.name] = item.value
    return result
