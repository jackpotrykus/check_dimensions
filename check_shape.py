from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, ParamSpec


T = TypeVar("T")
P = ParamSpec("P")


class IncompatibleDimensionError(Exception):
    pass


def create_dictionary_of_values_by_argument(f: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
    return {**{k: v for k, v in zip(f.__code__.co_varnames, args)}, **kwargs}



@dataclass
class Symbol(metaclass=ABCMeta):
    symbol: Any

    @abstractmethod
    def validate(self, axis_dimension: int, value_by_sym: dict[str, int]) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def create_error(self, axis_dimension: int, kwarg: str, value_by_sym: dict[str, int]) -> IncompatibleDimensionError:
        raise NotImplementedError
    
@dataclass
class Constant(Symbol):
    symbol: int

    def validate(self, axis_dimension: int, value_by_sym: dict[str, int]) -> bool:
        return self.symbol == axis_dimension
    
    def create_error(self, axis_dimension: int, kwarg: str, value_by_sym: dict[str, int]) -> IncompatibleDimensionError:
        return IncompatibleDimensionError(
            # f"Expected dimension {self.value} but got {axis_dimension}"
            f"Expected dimension {self.symbol} in {kwarg} to have dimension {self.symbol} but got {axis_dimension}"
        )
    
@dataclass
class Variable(Symbol):
    symbol: str

    def validate(self, axis_dimension: int, value_by_sym: dict[str, int]) -> bool:
        target_dim = value_by_sym.get(self.symbol, None)
        if target_dim is None:
            value_by_sym[self.symbol] = axis_dimension
            return True
        return axis_dimension == target_dim
                    
    def create_error(self, axis_dimension: int, kwarg: str, value_by_sym: dict[str, int]) -> IncompatibleDimensionError:
        return IncompatibleDimensionError(
            f"Expected dimension {self.symbol} in {kwarg} to have dimension {value_by_sym[self.symbol]} but got {axis_dimension}"
        )


def check_dimensions_against_spec(
    dimension_spec_by_kwarg: dict[str, str], dimension_by_kwarg: dict[str, tuple[int]]
) -> None:
    # unique_syms = get_unique_syms_from_dictionary_of_specs(dimension_spec_by_kwarg)
    # NOTE: Any constants (e.g. "3") will have empty values in this dictionary
    # Can easily be removed altogether but unnecessary also
    # value_by_sym = {sym: None for sym in unique_syms}
    value_by_sym = defaultdict(lambda: None)

    def parse_sym(sym: str) -> Symbol:
        try:
            int_sym = int(sym)
            return Constant(int_sym)
        except ValueError:
            return Variable(sym)

    for kwarg, dimensions in dimension_by_kwarg.items():
        for axis_dimension, sym in zip(dimensions, dimension_spec_by_kwarg[kwarg].split(",")):
            sym = parse_sym(sym)

            if not sym.validate(axis_dimension, value_by_sym):
                raise sym.create_error(axis_dimension, kwarg, value_by_sym)
            # Handle the case that axis dimension is specified exactly
            # try:
            #     int_sim = int(sym)
            #     if int_sim != axis_dimension:
            #         raise IncompatibleDimensionError(
            #             f"Expected dimension {sym} in {kwarg} to have dimension {int_sim} but got {axis_dimension}"
            #         )
            # except ValueError:
            #     # Handles the case that the axis dimension is a symbol
            #     if value_by_sym[sym] is None:
            #         value_by_sym[sym] = axis_dimension
            #     elif value_by_sym[sym] != axis_dimension:
            #         raise IncompatibleDimensionError(
            #             f"Expected dimension {sym} in {kwarg} to have dimension {value_by_sym[sym]} but got {axis_dimension}"
            #         )

    # TODO: obviously delete...
    print(value_by_sym)


def validate_dimension_spec_against_arguments(dimension_spec_by_kwarg: dict[str, str], value_by_kwarg: dict[str, Any]) -> None:
    for kwarg in dimension_spec_by_kwarg:
        if kwarg not in value_by_kwarg:
            raise KeyError(f"Expected keyword argument {kwarg} but it was not provided")

def check_shape(**dimension_spec_by_kwarg: dict[str, str]):
    # TODO: Obviously delete...
    # print(get_unique_syms_from_dictionary_of_specs(dimension_spec_by_kwarg))
    def decorator(f: Callable[P, T]) -> Callable[P, T]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            value_by_kwarg = create_dictionary_of_values_by_argument(f, *args, **kwargs)
            # Validate the supplied dimension spec against the supplied arguments
            validate_dimension_spec_against_arguments(dimension_spec_by_kwarg, value_by_kwarg)

            arraylike_by_kwarg = {k: value_by_kwarg[k] for k in dimension_spec_by_kwarg}
            dimension_by_kwarg = {k: tuple(int(i) for i in v.shape) for k, v in arraylike_by_kwarg.items()}
            check_dimensions_against_spec(dimension_spec_by_kwarg, dimension_by_kwarg)

            return f(*args, **kwargs)
        return wrapper
    return decorator

class CheckShape:
    def __call__(**dimension_spec_by_kwarg: dict[str, str]):
        return check_shape(**dimension_spec_by_kwarg)

