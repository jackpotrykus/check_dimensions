from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, ParamSpec, Protocol, Self, TypeVar

T = TypeVar("T")
P = ParamSpec("P")

RawAxisSpec = str | int
RawShapeSpec = tuple[RawAxisSpec, ...]
DimensionBySymbolDict = dict[str, int]


class IncompatibleShapeError(Exception):
    pass


class AxisSpec(Protocol):
    def validate(self, axis_dimension: int, dimension_by_symbol_spec: DimensionBySymbolDict) -> bool: ...

    def create_error(
        self, axis_dimension: int, kwarg: str, dimension_by_symbol_spec: DimensionBySymbolDict
    ) -> IncompatibleShapeError: ...


@dataclass(frozen=True)
class ConstantSpec(AxisSpec):
    spec: int

    def validate(self, axis_dimension: int, dimension_by_symbol_spec: DimensionBySymbolDict) -> bool:
        return self.spec == axis_dimension

    def create_error(
        self, axis_dimension: int, kwarg: str, dimension_by_symbol_spec: DimensionBySymbolDict 
    ) -> IncompatibleShapeError:
        return IncompatibleShapeError(
            f"Expected dimension {self.spec} in {kwarg} to have dimension {self.spec} but got {axis_dimension}"
        )


@dataclass(frozen=True)
class SymbolSpec(AxisSpec):
    spec: str

    def validate(self, axis_dimension: int, dimension_by_symbol_spec: DimensionBySymbolDict) -> bool:
        if self.spec not in dimension_by_symbol_spec:
            dimension_by_symbol_spec[self.spec] = axis_dimension
            return True

        target_dim = dimension_by_symbol_spec[self.spec]
        return axis_dimension == target_dim

    def create_error(
        self, axis_dimension: int, kwarg: str, dimension_by_symbol_spec: DimensionBySymbolDict
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


@dataclass
class ShapeSpec:
    axes: tuple[AxisSpec, ...]


RawShapeSpecByIdDict = dict[Hashable, RawShapeSpec]
ShapeSpecByIdDict = dict[Hashable, ShapeSpec]
ShapeByIdDict = dict[Hashable, tuple[int]]


@dataclass
class ShapeSpecCollection:
    shapes_by_id: ShapeSpecByIdDict
    dimension_by_symbol_spec: DimensionBySymbolDict = field(default_factory=dict)

    @classmethod
    def from_dict_of_raw_shape_specs(cls, raw_shape_specs_by_id: RawShapeSpecByIdDict) -> Self:
        def parse_raw_shape_spec_to_shape_spec(
            raw_shape_spec: RawShapeSpec,
        ) -> ShapeSpec:
            return ShapeSpec(tuple(parse_raw_axis_spec_to_axis_spec(axis) for axis in raw_shape_spec))

        shapes_by_id = {
            shape_id: parse_raw_shape_spec_to_shape_spec(raw_shape_spec)
            for shape_id, raw_shape_spec in raw_shape_specs_by_id.items()
        }
        return cls(shapes_by_id)

    def check_shapes(self, shape_by_id: ShapeByIdDict) -> None:
        def check_shape(shape_id: Hashable, shape: tuple[int]) -> bool:
            shape_spec = self.shapes_by_id[shape_id]
            if len(shape_spec.axes) != len(shape):
                return False

            for axis, axis_dimension in zip(shape_spec.axes, shape):
                if not axis.validate(axis_dimension, self.dimension_by_symbol_spec):
                    return False
            return True

        for shape_id, shape in shape_by_id.items():
            if not check_shape(shape_id, shape):
                raise IncompatibleShapeError(f"Shape {shape=} does not match {shape_id=}")


def get_shape_of_object(object: Any) -> tuple[int] | None:
    if hasattr(object, "shape"):
        return object.shape
    if hasattr(object, "__len__"):
        dims = [len(object)]
        while hasattr(object[0], "__len__"):
            object = object[0]
            dims.append(len(object))
        return tuple(dims)  # type: ignore
    # NOTE: Should this be an error?
    return None


@dataclass
class ShapeChecker:
    kwarg_shape_specs: ShapeSpecCollection | None = None
    return_shape_specs: ShapeSpecCollection | None = None

    def __call__(self, f: Callable[P, T]) -> Callable[P, T]:
        def create_dictionary_of_shapes_by_kwarg(f: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> ShapeByIdDict:
            d = {
                **{k: get_shape_of_object(v) for k, v in zip(f.__code__.co_varnames, args)},
                **kwargs,
            }
            return {k: v for k, v in d.items() if v is not None}

        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if self.kwarg_shape_specs is not None:
                shape_by_kwarg = create_dictionary_of_shapes_by_kwarg(f, *args, **kwargs)
                self.kwarg_shape_specs.check_shapes(shape_by_kwarg)

            res = f(*args, **kwargs)
            if self.return_shape_specs is not None:
                return_tuple = res if isinstance(res, tuple) else tuple([res])
                optional_shape_by_return_idx = {idx: get_shape_of_object(res) for idx, res in enumerate(return_tuple)}
                shape_by_return_idx: ShapeByIdDict = {
                    k: v for k, v in optional_shape_by_return_idx.items() if v is not None
                }
                self.return_shape_specs.check_shapes(shape_by_return_idx)
            return res

        return wrapper

    def args(self, **shape_spec_by_kwarg: RawShapeSpec) -> Self:
        self.kwarg_shape_specs = ShapeSpecCollection.from_dict_of_raw_shape_specs(shape_spec_by_kwarg)  # type: ignore
        return self

    def returns(self, *shape_spec: RawShapeSpec) -> Self:
        raw_spec_by_idx = {idx: raw_spec for idx, raw_spec in enumerate(shape_spec)}
        self.return_shape_specs = ShapeSpecCollection.from_dict_of_raw_shape_specs(raw_spec_by_idx)  # type: ignore
        return self


def args(**shape_spec_by_kwarg: RawShapeSpec) -> ShapeChecker:
    return ShapeChecker().args(**shape_spec_by_kwarg)


def returns(*shape_spec: RawShapeSpec) -> ShapeChecker:
    return ShapeChecker().returns(*shape_spec)
