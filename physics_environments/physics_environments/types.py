from dataclasses import dataclass, field, is_dataclass
from typing import Dict, Any
from abc import abstractmethod
import pprint

@dataclass
class ParamsBaseMixin:
    """ A class with common mixin methods for @dataclasses.
        It should be inherited like:

            @dataclass
            EnvConstants(ParamsBaseMixin)

                to add these methods.
    """
    def __getitem__(self, key):
        """ Allow dictionary indexing: params['n'] besides params.n.
            Consequently allows to convert to dict via __dict__ or dict(params).
        """
        return getattr(self, key)

    def to_dict(self):
        """ Fully convert the dataclass into a dictionary.
            While __dict__ works already, it does not convert entries itself into dicts.
            However a full dict representation can be very useful for getting a copy and paste for the params.

            Looks the best for outputs too!
        """
        def convert(value):
            if is_dataclass(value):
                return value.to_dict()  # Use a recursive algorithm
            # elif isinstance(value, list):  # handle lists
            #     return [convert(v) for v in value]
            # elif isinstance(value, dict):  # handle dictionaries
            #     return {k: convert(v) for k, v in value.items()}
            else:  # Return the value as is
                return value

        return {k: convert(v) for k, v in self.__dict__.items()}

    def __str__(self):
        """ Override the hard-to-read string representation from @dataclass, and return a clean dictionary """
        return self.to_dict().__str__() # the outputformat of the standart dict is better

    def keys(self):
        """ params.keys() method """
        return self.__dict__.keys()

    def values(self):
        """ params.values() method """
        return self.__dict__.values()

    def items(self):
        """ params.items() method """
        return self.__dict__.items()

    def __iter__(self):
        """ Allow 'for item_ in params: ...', etc. """
        return iter(vars(self))

    def __contains__(self, key):
        """ Support 'str' in params operations. """
        return key in vars(self)

    @abstractmethod
    def __post_init__(self):
        """ Useful for auto-defining with more complicated patterns. """
        pass