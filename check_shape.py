from typing import Any, Callable, TypeVar, ParamSpec
from pathlib import Path

import numpy as np


T = TypeVar("T")
P = ParamSpec("P")


class IncompatibleDimensionError(Exception):
    pass


def create_dictionary_of_values_by_argument(f: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
    return {**{k: v for k, v in zip(f.__code__.co_varnames, args)}, **kwargs}


def get_unique_syms_from_dictionary_of_specs(dimension_spec_by_kwarg: dict[str, str]) -> set[str]:
    return {sym.strip() for dimension_spec in dimension_spec_by_kwarg.values() for sym in dimension_spec.split(",")}


def check_dimensions_against_spec(
    dimension_spec_by_kwarg: dict[str, str], dimension_by_kwarg: dict[str, tuple[int]]
) -> None:
    unique_syms = get_unique_syms_from_dictionary_of_specs(dimension_spec_by_kwarg)
    # NOTE: Any constants (e.g. "3") will have empty values in this dictionary
    # Can easily be removed altogether but unnecessary also
    value_by_sym = {sym: None for sym in unique_syms}

    for kwarg, dimensions in dimension_by_kwarg.items():
        for axis_dimension, sym in zip(dimensions, dimension_spec_by_kwarg[kwarg].split(",")):
            sym = sym.strip()
            # Handle the case that axis dimension is specified exactly
            try:
                int_sim = int(sym)
                if int_sim != axis_dimension:
                    raise IncompatibleDimensionError(
                        f"Expected dimension {sym} in {kwarg} to have dimension {int_sim} but got {axis_dimension}"
                    )
            except ValueError:
                # Handles the case that the axis dimension is a symbol
                if value_by_sym[sym] is None:
                    value_by_sym[sym] = axis_dimension
                elif value_by_sym[sym] != axis_dimension:
                    raise IncompatibleDimensionError(
                        f"Expected dimension {sym} in {kwarg} to have dimension {value_by_sym[sym]} but got {axis_dimension}"
                    )

    # TODO: obviously delete...
    print(value_by_sym)


def validate_dimension_spec_against_arguments(dimension_spec_by_kwarg: dict[str, str], value_by_kwarg: dict[str, Any]) -> None:
    for kwarg in dimension_spec_by_kwarg:
        if kwarg not in value_by_kwarg:
            raise KeyError(f"Expected keyword argument {kwarg} but it was not provided")


def check_shape(**dimension_spec_by_kwarg: dict[str, str]) -> Callable[P, T]:
    # TODO: Obviously delete...
    print(get_unique_syms_from_dictionary_of_specs(dimension_spec_by_kwarg))
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
