__author__ = 'jack'
import warnings
from atom.api import Atom, Value, List, Str
import re
MODULE = re.compile(r"\w+(\.\w+)*$").match


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""
    def newFunc(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      # stacklevel=2,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc


def deprecate_module():
    warnings.warn("Use of deprecated module %s." % __name__,
                  # stacklevel=2,
                  category=DeprecationWarning)


class CustomEntryPoint(Atom):
    name = Str()
    module_name = Value()
    attrs = List()

    def load(self):
        entry = __import__(self.module_name, globals(), globals(), ['__name__'])
        for attr in self.attrs:
            try:
                entry = getattr(entry, attr)
            except AttributeError:
                raise ImportError("%r has no %r attribute" % (entry, attr))
        return entry

    @classmethod
    def parse(cls, key, value):
        """Parse a single entry point from string `src`

            key: some.module:some.attr

        The entry key and module name are required, but the ``:attrs`` part is optional
        """
        try:
            attrs = []
            if ':' in value:
                value, attrs = value.split(':', 1)
                if not MODULE(attrs.rstrip()):
                    raise ValueError
                attrs = attrs.rstrip().split('.')
        except ValueError:
            msg = "CustomEntryPoint must be in 'name=module:attrs' format"
            raise ValueError(msg, "%s=%s" % (key, value))
        else:
            return cls(name=key.strip(), module_name=value.strip(), attrs=attrs)

    @classmethod
    def instances_from_items(cls, items):
        result = {}
        for key, value in items:
            result[key] = cls.parse(key, value).load()
        return result
