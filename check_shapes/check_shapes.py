from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, ParamSpec, Protocol, Self, TypeVar
from collections.abc import Hashable

T = TypeVar("T")
P = ParamSpec("P")

Shape = tuple[int, ...]
RawAxisSpec = str | int
RawShapeSpec = tuple[RawAxisSpec, ...]
DimensionBySymbolDict = dict[str, int]


class IncompatibleShapeError(Exception):
    pass


# fmt: off
class AxisSpec(Protocol):
    def check_axis_dimension(self, axis_dimension: int, dimension_by_symbol: DimensionBySymbolDict) -> bool: ...
# fmt: on


@dataclass(frozen=True)
class ConstantSpec:
    spec: int

    def check_axis_dimension(self, axis_dimension: int, dimension_by_symbol: DimensionBySymbolDict) -> bool:
        return self.spec == axis_dimension


@dataclass(frozen=True)
class SymbolSpec:
    spec: str

    def check_axis_dimension(self, axis_dimension: int, dimension_by_symbol: DimensionBySymbolDict) -> bool:
        if self.spec not in dimension_by_symbol:
            dimension_by_symbol[self.spec] = axis_dimension
            return True

        target_dim = dimension_by_symbol[self.spec]
        return axis_dimension == target_dim


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


# NOTE: This class can maybe be deleted... I really thought I would do something with it
# but for now everything interesting happens in ShapeSpecCollection
@dataclass
class ShapeSpec:
    axes: tuple[AxisSpec, ...]


# yes, str is a Hashable. but this makes the type checker happy when passing in from **kwargs
RawShapeSpecByIdDict = dict[Hashable, RawShapeSpec]
ShapeSpecByIdDict = dict[Hashable, ShapeSpec]
OptionalShapeByIdDict = dict[Hashable, Shape | None]
ShapeByIdDict = dict[Hashable, Shape]


def parse_raw_shape_spec_to_shape_spec(
    raw_shape_spec: RawShapeSpec,
) -> ShapeSpec:
    return ShapeSpec(tuple(parse_raw_axis_spec_to_axis_spec(axis) for axis in raw_shape_spec))


def create_shapes_by_id_from_raw_shape_specs(raw_shape_specs_by_id: RawShapeSpecByIdDict) -> ShapeSpecByIdDict:
    return {
        shape_id: parse_raw_shape_spec_to_shape_spec(raw_shape_spec)
        for shape_id, raw_shape_spec in raw_shape_specs_by_id.items()
    }


from abc import ABCMeta, abstractmethod


@dataclass
class ShapeSpecCollection(metaclass=ABCMeta):
    shapes_by_id: ShapeSpecByIdDict = field(default_factory=dict)
    dimension_by_symbol: DimensionBySymbolDict = field(default_factory=dict)

    @classmethod
    def from_dict_of_identifiers(cls, raw_shape_specs_by_id: RawShapeSpecByIdDict) -> Self:
        shapes_by_id = create_shapes_by_id_from_raw_shape_specs(raw_shape_specs_by_id)
        return cls(shapes_by_id)

    def add(self, **raw_shape_specs_by_id: RawShapeSpec) -> Self:
        shapes_by_id = create_shapes_by_id_from_raw_shape_specs(raw_shape_specs_by_id)  # type: ignore
        self.shapes_by_id = {**self.shapes_by_id, **shapes_by_id}
        return self

    def check_shapes(self, shape_by_id: ShapeByIdDict) -> None:
        def check_shape(shape_id: Hashable, shape: Shape) -> bool:
            # shape_spec = self.shapes_by_id[shape_id]
            shape_spec = self.shapes_by_id.get(shape_id)
            # TODO: This is right... right??
            if shape_spec is None:
                return True

            if len(shape_spec.axes) != len(shape):
                return False

            for axis, axis_dimension in zip(shape_spec.axes, shape):
                if not axis.check_axis_dimension(axis_dimension, self.dimension_by_symbol):
                    return False
            return True

        for shape_id, shape in shape_by_id.items():
            if not check_shape(shape_id, shape):
                raise IncompatibleShapeError(f"Shape {shape=} does not match {shape_id=}")


class ArgSpecCollection(ShapeSpecCollection):
    pass


class ReturnSpecCollection(ShapeSpecCollection):
    pass


def get_shape_of_object(obj: Any) -> Shape | None:
    if hasattr(obj, "shape"):
        return obj.shape
    if isinstance(obj, (list, tuple)):
        return (len(obj),)
    return None


@dataclass
class CheckShapesFunctionDecorator:
    arg_shape_specs: ArgSpecCollection = field(default_factory=ArgSpecCollection)
    return_shape_specs: ReturnSpecCollection = field(default_factory=ReturnSpecCollection)

    def __call__(self, f: Callable[P, T]) -> Callable[P, T]:
        def create_dict_by_arg(f: Callable[P, T], *args: P.args, **_: P.kwargs) -> dict[str, Any]:
            return {k: v for k, v in zip(f.__code__.co_varnames, args)}

        def create_dict_of_shapes_by_arg(f: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> ShapeByIdDict:
            d1: OptionalShapeByIdDict = create_dict_by_arg(f, *[get_shape_of_object(v) for v in args])  # type: ignore
            d2: OptionalShapeByIdDict = {k: get_shape_of_object(v) for k, v in kwargs.items()}
            return {k: v for k, v in {**d1, **d2}.items() if v is not None}

        def update_shapes_by_id(shapes_by_id: ShapeByIdDict, f: Callable[..., Any]) -> ShapeByIdDict:
            int_keys = {k: v for k, v in shapes_by_id.items() if isinstance(k, int)}
            str_keys = {k: v for k, v in shapes_by_id.items() if isinstance(k, str)}
            updated_shapes_by_id = {
                **shapes_by_id,
                **create_dict_by_arg(f, *[v for k, v in int_keys.items()]),  # type: ignore
                **str_keys,
            }
            return updated_shapes_by_id

        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # TODO: Cleanup this mess
            shape_by_arg = create_dict_of_shapes_by_arg(f, *args, **kwargs)
            print("by arg", shape_by_arg)
            print("by id", self.arg_shape_specs.shapes_by_id)

            self.arg_shape_specs.shapes_by_id = update_shapes_by_id(self.arg_shape_specs.shapes_by_id, f)  # type: ignore

            print("by id", self.arg_shape_specs.shapes_by_id)
            self.arg_shape_specs.check_shapes(shape_by_arg)  # type: ignore

            res = f(*args, **kwargs)
            return_tuple = res if isinstance(res, tuple) else (res,)
            optional_shape_by_return_idx = {idx: get_shape_of_object(r) for idx, r in enumerate(return_tuple)}
            shape_by_return_idx: ShapeByIdDict = {
                k: v for k, v in optional_shape_by_return_idx.items() if v is not None
            }
            self.return_shape_specs.check_shapes(shape_by_return_idx)
            return res

        return wrapper

    def args(self, *shape_spec: RawShapeSpec, **shape_spec_by_kwarg: RawShapeSpec) -> Self:
        raw_spec_by_idx = {idx: raw_spec for idx, raw_spec in enumerate(shape_spec)}
        self.arg_shape_specs = self.arg_shape_specs.from_dict_of_identifiers(raw_spec_by_idx)  # type: ignore
        return self.kwargs(**shape_spec_by_kwarg)

    def kwargs(self, **shape_spec_by_kwarg: RawShapeSpec) -> Self:
        self.arg_shape_specs = self.arg_shape_specs.add(**shape_spec_by_kwarg)  # type: ignore
        return self

    def returns(self, *shape_spec: RawShapeSpec) -> Self:
        raw_spec_by_idx = {idx: raw_spec for idx, raw_spec in enumerate(shape_spec)}
        self.return_shape_specs = ReturnSpecCollection.from_dict_of_identifiers(raw_spec_by_idx)  # type: ignore
        return self


def args(*shape_spec: RawShapeSpec, **shape_spec_by_kwarg: RawShapeSpec) -> CheckShapesFunctionDecorator:
    return CheckShapesFunctionDecorator().args(*shape_spec).kwargs(**shape_spec_by_kwarg)


def kwargs(**shape_spec_by_kwarg: RawShapeSpec) -> CheckShapesFunctionDecorator:
    return CheckShapesFunctionDecorator().kwargs(**shape_spec_by_kwarg)


def returns(*shape_spec: RawShapeSpec) -> CheckShapesFunctionDecorator:
    return CheckShapesFunctionDecorator().returns(*shape_spec)
