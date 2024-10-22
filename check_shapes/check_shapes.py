from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, ParamSpec, Protocol, Self, TypeVar

T = TypeVar("T")
P = ParamSpec("P")

RawAxisSpec = str | int
RawShapeSpec = tuple[RawAxisSpec, ...]
DimensionBySymbolDict = dict[str, int]


class IncompatibleShapeError(Exception):
    pass


# fmt: off
class AxisSpec(Protocol):
    def check_axis_dimension(self, axis_dimension: int, dimension_by_symbol: DimensionBySymbolDict) -> bool: ...
    def create_error(self, axis_dimension: int, kwarg: str, dimension_by_symbol: DimensionBySymbolDict) -> IncompatibleShapeError: ...
# fmt: on


@dataclass(frozen=True)
class ConstantSpec:
    spec: int

    def check_axis_dimension(self, axis_dimension: int, dimension_by_symbol: DimensionBySymbolDict) -> bool:
        return self.spec == axis_dimension

    def create_error(
        self, axis_dimension: int, kwarg: str, dimension_by_symbol: DimensionBySymbolDict
    ) -> IncompatibleShapeError:
        return IncompatibleShapeError(
            f"Expected dimension {self.spec} in {kwarg} to have dimension {self.spec} but got {axis_dimension}"
        )


@dataclass(frozen=True)
class SymbolSpec:
    spec: str

    def check_axis_dimension(self, axis_dimension: int, dimension_by_symbol: DimensionBySymbolDict) -> bool:
        if self.spec not in dimension_by_symbol:
            dimension_by_symbol[self.spec] = axis_dimension
            return True

        target_dim = dimension_by_symbol[self.spec]
        return axis_dimension == target_dim

    def create_error(
        self, axis_dimension: int, kwarg: str, dimension_by_symbol: DimensionBySymbolDict
    ) -> IncompatibleShapeError:
        return IncompatibleShapeError(
            f"Expected dimension {self.spec} in {kwarg} to have dimension {dimension_by_symbol[self.spec]} but got {axis_dimension}"
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


# NOTE: This class can maybe be deleted... I really thought I would do something with it
# but for now everything interesting happens in ShapeSpecCollection
@dataclass
class ShapeSpec:
    axes: tuple[AxisSpec, ...]


# yes, str is a Hashable. but this makes the type checker happy when passing in from **kwargs
RawShapeSpecByIdDict = dict[Hashable, RawShapeSpec]
ShapeSpecByIdDict = dict[Hashable, ShapeSpec]
ShapeByIdDict = dict[Hashable, tuple[int]]

def parse_raw_shape_spec_to_shape_spec(
    raw_shape_spec: RawShapeSpec,
) -> ShapeSpec:
    return ShapeSpec(tuple(parse_raw_axis_spec_to_axis_spec(axis) for axis in raw_shape_spec))


def create_shapes_by_id_from_raw_shape_specs(raw_shape_specs_by_id: RawShapeSpecByIdDict) -> ShapeSpecByIdDict:
    return {shape_id: parse_raw_shape_spec_to_shape_spec(raw_shape_spec) for shape_id, raw_shape_spec in raw_shape_specs_by_id.items()}


@dataclass
class ShapeSpecCollection:
    shapes_by_id: ShapeSpecByIdDict
    dimension_by_symbol: DimensionBySymbolDict = field(default_factory=dict)

    @classmethod
    def from_dict_of_raw_shape_specs(cls, raw_shape_specs_by_id: RawShapeSpecByIdDict) -> Self:
        shapes_by_id = create_shapes_by_id_from_raw_shape_specs(raw_shape_specs_by_id)
        return cls(shapes_by_id)
    
    def add(self, **raw_shape_specs_by_id: RawShapeSpec) -> Self:
        shapes_by_id = create_shapes_by_id_from_raw_shape_specs(raw_shape_specs_by_id)  # type: ignore
        self.shapes_by_id = {**self.shapes_by_id, **shapes_by_id}
        return self

    def check_shapes(self, shape_by_id: ShapeByIdDict) -> None:
        def check_shape(shape_id: Hashable, shape: tuple[int]) -> bool:
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


def get_shape_of_object(object: Any) -> tuple[int] | None:
    if hasattr(object, "shape"):
        return object.shape
    if hasattr(object, "__len__"):
        dims = [len(object)]
        while hasattr(object[0], "__len__"):
            object = object[0]
            dims.append(len(object))
        return tuple(dims)  # type: ignore
    return None


@dataclass
class FunctionChecker:
    arg_shape_specs: ShapeSpecCollection | None = None
    return_shape_specs: ShapeSpecCollection | None = None

    def __call__(self, f: Callable[P, T]) -> Callable[P, T]:
        def create_dict_by_arg(f: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
            return {**{k: v for k, v in zip(f.__code__.co_varnames, args)}, **kwargs}

        def create_dict_of_shapes_by_arg(f: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> ShapeByIdDict:
            d = create_dict_by_arg(
                f,
                *[get_shape_of_object(v) for v in args],
                **{k: get_shape_of_object(v) for k, v in kwargs.items()}
            )  # type: ignore
            print("d", d)
            return {k: v for k, v in d.items() if v is not None}

        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if self.arg_shape_specs is not None:
                shape_by_arg = create_dict_of_shapes_by_arg(f, *args, **kwargs)
                print("by arg", shape_by_arg)
                print("by id", self.arg_shape_specs.shapes_by_id)
                self.arg_shape_specs.shapes_by_id = create_dict_by_arg(
                    f, 
                    *[v for k, v in self.arg_shape_specs.shapes_by_id.items() if isinstance(k, int)], 
                    **{k: v for k, v in self.arg_shape_specs.shapes_by_id.items() if isinstance(k, str)},
                )  # type: ignore
                print("by id", self.arg_shape_specs.shapes_by_id)
                self.arg_shape_specs.check_shapes(shape_by_arg)  # type: ignore

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

    def args(self, *shape_spec: RawShapeSpec, **shape_spec_by_kwarg: RawShapeSpec) -> Self:
        raw_spec_by_idx = {idx: raw_spec for idx, raw_spec in enumerate(shape_spec)}
        print("raw spec by idx!")
        print(raw_spec_by_idx)
        self.arg_shape_specs = ShapeSpecCollection.from_dict_of_raw_shape_specs(raw_spec_by_idx)  # type: ignore
        self.arg_shape_specs = self.arg_shape_specs.add(**shape_spec_by_kwarg)
        print(self.arg_shape_specs)
        return self

    def kwargs(self, **shape_spec_by_kwarg: RawShapeSpec) -> Self:
        self.arg_shape_specs = ShapeSpecCollection.from_dict_of_raw_shape_specs(shape_spec_by_kwarg)  # type: ignore
        return self

    def returns(self, *shape_spec: RawShapeSpec) -> Self:
        raw_spec_by_idx = {idx: raw_spec for idx, raw_spec in enumerate(shape_spec)}
        self.return_shape_specs = ShapeSpecCollection.from_dict_of_raw_shape_specs(raw_spec_by_idx)  # type: ignore
        return self


def args(*shape_spec: RawShapeSpec, **shape_spec_by_kwarg: RawShapeSpec) -> FunctionChecker:
    return FunctionChecker().args(*shape_spec, **shape_spec_by_kwarg)#.kwargs(**shape_spec_by_kwarg)


def kwargs(**shape_spec_by_kwarg: RawShapeSpec) -> FunctionChecker:
    return FunctionChecker().kwargs(**shape_spec_by_kwarg)


def returns(*shape_spec: RawShapeSpec) -> FunctionChecker:
    return FunctionChecker().returns(*shape_spec)
