from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, ParamSpec, Self


T = TypeVar("T")
P = ParamSpec("P")

RawAxisSpec = str | int
RawShapeSpec = tuple[RawAxisSpec, ...]


class IncompatibleShapeError(Exception):
    pass


@dataclass(frozen=True)
class AxisSpec(metaclass=ABCMeta):
    spec: Any

    @abstractmethod
    def validate(self, axis_dimension: int, dimension_by_symbol_spec: dict[str, int]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def create_error(
        self, axis_dimension: int, kwarg: str, dimension_by_symbol_spec: dict[str, int]
    ) -> IncompatibleShapeError:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantSpec(AxisSpec):
    spec: int

    def validate(self, axis_dimension: int, dimension_by_symbol_spec: dict[str, int]) -> bool:
        return self.spec == axis_dimension

    def create_error(
        self, axis_dimension: int, kwarg: str, dimension_by_symbol_spec: dict[str, int]
    ) -> IncompatibleShapeError:
        return IncompatibleShapeError(
            # f"Expected dimension {self.value} but got {axis_dimension}"
            f"Expected dimension {self.spec} in {kwarg} to have dimension {self.spec} but got {axis_dimension}"
        )


@dataclass(frozen=True)
class SymbolSpec(AxisSpec):
    spec: str

    def validate(self, axis_dimension: int, dimension_by_symbol_spec: dict[str, int]) -> bool:
        if self.spec not in dimension_by_symbol_spec:
            dimension_by_symbol_spec[self.spec] = axis_dimension
            return True

        target_dim = dimension_by_symbol_spec[self.spec]
        return axis_dimension == target_dim

    def create_error(
        self, axis_dimension: int, kwarg: str, dimension_by_symbol_spec: dict[str, int]
    ) -> IncompatibleShapeError:
        return IncompatibleShapeError(
            f"Expected dimension {self.spec} in {kwarg} to have dimension {dimension_by_symbol_spec[self.spec]} but got {axis_dimension}"
        )


def parse_raw_axis_spec_to_axis_spec(symbol: RawAxisSpec) -> AxisSpec:
    spec_by_type = {
        int: ConstantSpec,
        str: SymbolSpec,
    }
    for type_, spec in spec_by_type.items():
        if isinstance(symbol, type_):
            return spec(symbol)
    raise ValueError(
        f"Cannot parse axis length specifier {symbol}, expected one of {','.join(str(s) for s in spec_by_type.keys())}"
    )


def check_kwarg_shapes_against_spec(
    shape_spec_by_kwarg: dict[str, RawShapeSpec], shape_by_kwarg: dict[str, tuple[int]]
) -> None:
    dimension_by_symbol_spec: dict[str, int] = {}

    def validate_shape_spec(shape_spec: RawShapeSpec, shape: tuple[int]) -> None:
        if len(shape_spec) != len(shape):
            raise IncompatibleShapeError(f"Length of {shape_spec=} does not match length of {shape=}")

        def parse_raw_shape_spec_to_list_axis_spec(shape_spec: RawShapeSpec) -> list[AxisSpec]:
            return [parse_raw_axis_spec_to_axis_spec(axis) for axis in shape_spec]

        for axis_dimension, axis_spec in zip(shape, parse_raw_shape_spec_to_list_axis_spec(shape_spec)):
            if not axis_spec.validate(axis_dimension, dimension_by_symbol_spec):
                raise axis_spec.create_error(axis_dimension, kwarg, dimension_by_symbol_spec)

    for kwarg, shape in shape_by_kwarg.items():
        raw_shape_spec = shape_spec_by_kwarg[kwarg]
        validate_shape_spec(raw_shape_spec, shape)

    # TODO: obviously delete...
    print(dimension_by_symbol_spec)


def validate_shape_spec(shape_spec_by_kwarg: dict[str, RawShapeSpec], value_by_kwarg: dict[str, Any]) -> None:
    for kwarg in shape_spec_by_kwarg:
        if kwarg not in value_by_kwarg:
            raise KeyError(f"Expected keyword argument {kwarg} but it was not provided")


def get_shape_of_object(object: Any) -> tuple[int]:
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
    f: Callable[P, T], shape_spec_by_kwarg: dict[str, RawShapeSpec] | None, return_spec: RawShapeSpec | None
) -> Callable[P, T]:
    def create_dictionary_of_values_by_kwarg(f: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
        return {**{k: v for k, v in zip(f.__code__.co_varnames, args)}, **kwargs}

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if shape_spec_by_kwarg is not None:
            value_by_kwarg = create_dictionary_of_values_by_kwarg(f, *args, **kwargs)
            # Validate the supplied shape spec against the supplied arguments
            validate_shape_spec(shape_spec_by_kwarg, value_by_kwarg)

            arraylike_by_kwarg = {k: value_by_kwarg[k] for k in shape_spec_by_kwarg}
            shape_by_kwarg = {k: get_shape_of_object(v) for k, v in arraylike_by_kwarg.items()}
            check_kwarg_shapes_against_spec(shape_spec_by_kwarg, shape_by_kwarg)

        res = f(*args, **kwargs)
        if return_spec is not None:
            validate_shape_spec({"return": return_spec}, {"return": get_shape_of_object(res)})
        return res

    return wrapper


@dataclass
class ShapeChecker:
    shape_spec_by_kwarg: dict[str, RawShapeSpec] | None = None
    return_spec: RawShapeSpec | None = None

    def args(self, **shape_spec_by_kwarg: RawShapeSpec) -> Self:
        self.shape_spec_by_kwarg = shape_spec_by_kwarg
        return self

    def returns(self, shape_spec: RawShapeSpec) -> Self:
        self.return_spec = shape_spec
        return self

    def __call__(self, f: Callable[P, T]) -> Callable[P, T]:
        return _check_shapes(f, self.shape_spec_by_kwarg, self.return_spec)


def args(**shape_spec_by_kwarg: RawShapeSpec) -> ShapeChecker:
    return ShapeChecker(shape_spec_by_kwarg=shape_spec_by_kwarg)


def returns(shape_spec: RawShapeSpec) -> ShapeChecker:
    return ShapeChecker(return_spec=shape_spec)
