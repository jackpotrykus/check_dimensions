from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, ParamSpec, Self


T = TypeVar("T")
P = ParamSpec("P")

AxisLengthSpecifier = str | int
ShapeSpecifier = tuple[AxisLengthSpecifier, ...]


class IncompatibleDimensionError(Exception):
    pass


def create_dictionary_of_values_by_kwarg(f: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
    return {**{k: v for k, v in zip(f.__code__.co_varnames, args)}, **kwargs}


@dataclass
class AxisSpec(metaclass=ABCMeta):
    spec: Any

    @abstractmethod
    def validate(self, axis_dimension: int, value_by_sym: dict[str, int]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def create_error(self, axis_dimension: int, kwarg: str, value_by_sym: dict[str, int]) -> IncompatibleDimensionError:
        raise NotImplementedError


@dataclass
class ConstantSpec(AxisSpec):
    spec: int

    def validate(self, axis_dimension: int, value_by_sym: dict[str, int]) -> bool:
        return self.spec == axis_dimension

    def create_error(self, axis_dimension: int, kwarg: str, value_by_sym: dict[str, int]) -> IncompatibleDimensionError:
        return IncompatibleDimensionError(
            # f"Expected dimension {self.value} but got {axis_dimension}"
            f"Expected dimension {self.spec} in {kwarg} to have dimension {self.spec} but got {axis_dimension}"
        )


@dataclass
class SymbolSpec(AxisSpec):
    spec: str

    def validate(self, axis_dimension: int, value_by_sym: dict[str, int]) -> bool:
        if self.spec not in value_by_sym:
            value_by_sym[self.spec] = axis_dimension
            return True

        target_dim = value_by_sym[self.spec]
        return axis_dimension == target_dim

    def create_error(self, axis_dimension: int, kwarg: str, value_by_sym: dict[str, int]) -> IncompatibleDimensionError:
        return IncompatibleDimensionError(
            f"Expected dimension {self.spec} in {kwarg} to have dimension {value_by_sym[self.spec]} but got {axis_dimension}"
        )

def parse_axis_length_specifier_to_axis_spec(symbol: AxisLengthSpecifier) -> AxisSpec:
    spec_by_type = {
        int: ConstantSpec,
        str: SymbolSpec,
    }
    for type_, spec in spec_by_type.items():
        if isinstance(symbol, type_):
            return spec(symbol)
    raise ValueError(f"Cannot parse axis length specifier {symbol}, expected one of {','.join(str(s) for s in spec_by_type.keys())}")

def check_kwarg_dimensions_against_spec(
    dimension_spec_by_kwarg: dict[str, ShapeSpecifier], dimension_by_kwarg: dict[str, tuple[int]]
) -> None:
    value_by_sym: dict[str, int] = {}
        
    def create_list_of_specs(dimension_spec: ShapeSpecifier) -> list[AxisSpec]:
        return [parse_axis_length_specifier_to_axis_spec(spec) for spec in dimension_spec]

    for kwarg, dimensions in dimension_by_kwarg.items():
        for axis_dimension, spec in zip(dimensions, create_list_of_specs(dimension_spec_by_kwarg[kwarg])):
            if not spec.validate(axis_dimension, value_by_sym):
                raise spec.create_error(axis_dimension, kwarg, value_by_sym)

    # TODO: obviously delete...
    print(value_by_sym)


def validate_dimension_spec(dimension_spec_by_kwarg: dict[str, ShapeSpecifier], value_by_kwarg: dict[str, Any]) -> None:
    for kwarg in dimension_spec_by_kwarg:
        if kwarg not in value_by_kwarg:
            raise KeyError(f"Expected keyword argument {kwarg} but it was not provided")


def parse_shape(object: Any) -> tuple[int]:
    if hasattr(object, "shape"):
        return object.shape
    if hasattr(object, "__len__"):
        dims = [len(object)]
        while hasattr(object[0], "__len__"):
            object = object[0]
            dims.append(len(object))
        return tuple(dims)  # type: ignore
    raise ValueError(f"Cannot parse shape of {object}")


def _check_shapes(
    f: Callable[P, T], dimension_spec_by_kwarg: dict[str, ShapeSpecifier] | None, return_spec: ShapeSpecifier | None
) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if dimension_spec_by_kwarg is not None:
            value_by_kwarg = create_dictionary_of_values_by_kwarg(f, *args, **kwargs)
            # Validate the supplied dimension spec against the supplied arguments
            validate_dimension_spec(dimension_spec_by_kwarg, value_by_kwarg)

            arraylike_by_kwarg = {k: value_by_kwarg[k] for k in dimension_spec_by_kwarg}
            dimension_by_kwarg = {k: parse_shape(v) for k, v in arraylike_by_kwarg.items()}
            check_kwarg_dimensions_against_spec(dimension_spec_by_kwarg, dimension_by_kwarg)

        res = f(*args, **kwargs)
        if return_spec is not None:
            validate_dimension_spec({"return": return_spec}, {"return": parse_shape(res)})
        return res

    return wrapper


@dataclass
class ShapeChecker:
    dimension_spec_by_kwarg: dict[str, ShapeSpecifier] | None = None
    return_spec: ShapeSpecifier | None = None

    def args(self, **dimension_spec_by_kwarg: ShapeSpecifier) -> Self:
        self.dimension_spec_by_kwarg = dimension_spec_by_kwarg
        return self

    def returns(self, dimension_spec: ShapeSpecifier) -> Self:
        self.return_spec = dimension_spec
        return self

    def __call__(self, f: Callable[P, T]) -> Callable[P, T]:
        return _check_shapes(f, self.dimension_spec_by_kwarg, self.return_spec)


def args(**dimension_spec_by_kwarg: ShapeSpecifier) -> ShapeChecker:
    return ShapeChecker(dimension_spec_by_kwarg=dimension_spec_by_kwarg)


def returns(dimension_spec: ShapeSpecifier) -> ShapeChecker:
    return ShapeChecker(return_spec=dimension_spec)
