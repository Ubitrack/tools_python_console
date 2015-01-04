import os
from atom.api import Atom, Typed, List, Dict, Enum, Str

import logging

from .utql_pattern_parser import parse as parse_utql_pattern, find_all_patterns
from .base_parser import UTQL_NAMESPACES
from .utql_request_parser import parse as parse_utql_request

log = logging.getLogger(__name__)


class RequestLoader(Atom):

    pattern_directory = Str()
    namespaces = Dict(UTQL_NAMESPACES)

    pattern_templates = Dict()

    def _default_pattern_templates(self):
        if not os.path.isdir(self.pattern_directory):
            log.error("Invalid pattern directory: %s" % self.pattern_directory)
            return {}

        pattern_templates = {}
        for filename in find_all_patterns(root_path=os.path.abspath(self.pattern_directory)):
            log.info("Parsing pattern-file: %s" % filename)
            result = parse_utql_pattern(filename, self.namespaces)
            if result is not None:
                for k, v in result.items():
                    pattern_templates[k] = v

        return pattern_templates

    def load_request(self, filename):
        return parse_utql_request(filename, self.namespaces)

    def process_request(self, request):
        template_keys = self.pattern_templates.keys()
        for pattern in request.patterns:
            if pattern.name not in template_keys:
                log.warning("Pattern: %s not in template library - skipping...")
                continue
            template = self.pattern_templates[pattern.name]

