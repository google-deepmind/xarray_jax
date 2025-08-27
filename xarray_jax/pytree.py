# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""JAX PyTree registration logic."""

import collections
import contextlib
import contextvars
from typing import Any, Callable, Iterator, Mapping, Optional, Union, Tuple, cast
from typing import Hashable  # pylint: disable=deprecated-class

import jax
import jax.numpy as jnp
import numpy as np
import xarray
from xarray_jax import core


def wrap(value):
  """Deprecated identity function for backward compatibility."""
  return value


def unwrap(value, require_jax=False):
  """Unwraps NonArrayLeafWrapper instances, passing through other values."""
  unwrapped = value.leaf if isinstance(value, NonArrayLeafWrapper) else value
  if require_jax and not isinstance(unwrapped, jax.Array):
    raise TypeError(f'Expected JAX array, found {type(unwrapped)}.')
  return unwrapped


def unwrap_data(
    value: Union[xarray.Variable, xarray.DataArray],
    require_jax: bool = False
    ) -> Union[jax.Array, np.ndarray]:
  """The unwrapped (see unwrap) data of a an xarray.Variable or DataArray."""
  return unwrap(value.data, require_jax=require_jax)


def unwrap_vars(
    dataset: Mapping[Hashable, xarray.DataArray],
    require_jax: bool = False
    ) -> Mapping[str, Union[jax.Array, np.ndarray]]:
  """The unwrapped data (see unwrap) of the variables in a dataset."""
  # xarray types variable names as Hashable, but in practice they're invariably
  # strings and we convert to str to allow for a more useful return type.
  return {str(name): unwrap_data(var, require_jax=require_jax)
          for name, var in dataset.items()}


def unwrap_coords(
    dataset: Union[xarray.Dataset, xarray.DataArray],
    require_jax: bool = False
    ) -> Mapping[str, Union[jax.Array, np.ndarray]]:
  """The unwrapped data (see unwrap) of the coords in a Dataset or DataArray."""
  return {str(name): unwrap_data(var, require_jax=require_jax)
          for name, var in dataset.coords.items()}


def jax_data(value: Union[xarray.Variable, xarray.DataArray]) -> jax.Array:
  """Like unwrap_data, but will complain if not a jax array."""
  # Implementing this separately so we can give a more specific return type
  # for it.
  return cast(jax.Array, unwrap_data(value, require_jax=True))


def jax_vars(
    dataset: Mapping[Hashable, xarray.DataArray]) -> Mapping[str, jax.Array]:
  """Like unwrap_vars, but will complain if vars are not all jax arrays."""
  return cast(Mapping[str, jax.Array], unwrap_vars(dataset, require_jax=True))


class NonArrayLeafWrapper:
  """Wraps non-array leaf value into a duck-typed array for use with xarray.

  This is necessary to satisfy the JAX contract and handle cases where JAX
  transformations produce non-array leaf values (e.g., Python scalars,
  ShapeDtypeStruct) that must be re-inserted into an xarray structure.

  Provides a minimal array-like interface required by xarray's constructors and
  raises TypeError for unsupported operations on non-array data types.
  """

  def __init__(self, leaf: Any, dims: Tuple[Hashable, ...]):
    self._leaf = leaf
    self._dims = dims

    # Use a zero-sized shape for non-array data to indicate that this is a
    # placeholder
    self._shape = getattr(leaf, 'shape', (0,) * len(dims))

    # Determine dtype.
    if hasattr(leaf, 'dtype'):
      self._dtype = leaf.dtype
    elif isinstance(leaf, bool):
      self._dtype = jnp.bool_.dtype
    elif isinstance(leaf, int):
      self._dtype = jnp.int32.dtype
    elif isinstance(leaf, float):
      self._dtype = jnp.float32.dtype
    else:
      self._dtype = np.dtype(object)

  # TODO(matthjw): Migrate to the Python Array API standard
  def __array_ufunc__(self, ufunc, method, *args, **kwargs):
    raise TypeError(
        f"NumPy ufunc '{ufunc.__name__}' is not supported on non-array JAX "
        f'leaf of type {type(self._leaf).__name__}.'
    )

  def __array_function__(self, func, types, args, kwargs):
    raise TypeError(
        f"NumPy function '{func.__name__}' is not supported on non-array JAX "
        f'leaf of type {type(self._leaf).__name__}.'
    )

  @property
  def shape(self):
    return self._shape

  @property
  def dtype(self):
    return self._dtype

  @property
  def ndim(self):
    return len(self._dims)

  @property
  def size(self):
    return np.prod(self._shape)

  def __getitem__(self, key):
    raise TypeError(
        f'Indexing is not supported on non-array leaf of type '
        f'{type(self._leaf).__name__}.'
    )

  @property
  def leaf(self):
    """Provides access to the original, unwrapped leaf."""
    return self._leaf


# Register xarray datatypes with jax.tree_util.


DimsChangeFn = Callable[[Tuple[Hashable, ...]], Tuple[Hashable, ...]]
_DIMS_CHANGE_ON_UNFLATTEN_FN: contextvars.ContextVar[DimsChangeFn] = (
    contextvars.ContextVar('dims_change_on_unflatten_fn'))


@contextlib.contextmanager
def dims_change_on_unflatten(dims_change_fn: DimsChangeFn):
  """Can be used to change the dims used when unflattening arrays into xarrays.

  This is useful when some axes were added to / removed from the underlying jax
  arrays after they were flattened using jax.tree_util.tree_flatten, and you
  want to unflatten them again afterwards using the original treedef but
  adjusted for the added/removed dimensions.

  It can also be used with jax.tree_util.tree_map, when it's called with a
  function that adds/removes axes or otherwise changes the axis order.

  When dimensions are removed, any coordinates using those removed dimensions
  will also be removed on unflatten.

  This is implemented as a context manager that sets some thread-local state
  affecting the behaviour of our unflatten functions, because it's not possible
  to directly modify the treedef to change the dims/coords in it (and with
  tree_map, the treedef isn't exposed to you anyway).

  Args:
    dims_change_fn: Maps a tuple of dimension names for the original
      Variable/DataArray/Dataset that was flattened, to an updated tuple of
      dimensions which should be used when unflattening.

  Yields:
    To a context manager in whose scope jax.tree_util.tree_unflatten and
    jax.tree_util.tree_map will apply the dims_change_fn before reconstructing
    xarrays from jax arrays.
  """
  token = _DIMS_CHANGE_ON_UNFLATTEN_FN.set(dims_change_fn)
  try:
    yield
  finally:
    _DIMS_CHANGE_ON_UNFLATTEN_FN.reset(token)


def _flatten_variable(v: xarray.Variable) -> Tuple[
    Tuple[Any], Tuple[Hashable, ...]]:  # pylint: disable=g-one-element-tuple
  """Flattens a Variable for jax.tree_util."""
  children = (unwrap_data(v),)
  aux = v.dims
  return children, aux


def _unflatten_variable(
    aux: Tuple[Hashable, ...],
    children: Tuple[Any]) -> xarray.Variable:  # pylint: disable=g-one-element-tuple
  """Unflattens a Variable for jax.tree_util."""
  dims = aux
  data = children[0]

  dims_change_fn = _DIMS_CHANGE_ON_UNFLATTEN_FN.get(None)
  if dims_change_fn: dims = dims_change_fn(dims)

  if isinstance(data, (jax.Array, np.ndarray)):
    return xarray.Variable(dims=dims, data=data)
  else:
    wrapper = NonArrayLeafWrapper(leaf=data, dims=dims)
    return xarray.Variable(dims=dims, data=wrapper)


def _split_static_and_jax_coords(
    coords: xarray.core.coordinates.Coordinates) -> Tuple[
        Mapping[Hashable, xarray.Variable], Mapping[Hashable, xarray.Variable]]:
  static_coord_vars = {}
  jax_coord_vars = {}
  for name, coord in coords.items():
    if coord.attrs.get(core.JAX_COORD_ATTR_NAME, False):
      jax_coord_vars[name] = coord.variable
    else:
      assert not isinstance(coord, (jax.Array, NonArrayLeafWrapper))
      static_coord_vars[name] = coord.variable
  return static_coord_vars, jax_coord_vars


def _drop_with_none_of_dims(
    coord_vars: Mapping[Hashable, xarray.Variable],
    dims: Tuple[Hashable, ...]) -> Mapping[Hashable, xarray.Variable]:
  return {name: var for name, var in coord_vars.items()
          if set(var.dims) <= set(dims)}


class _HashableCoords(collections.abc.Mapping):
  """Wraps a dict of xarray Variables as hashable, used for static coordinates.

  This needs to be hashable so that when an xarray.Dataset is passed to a
  jax.jit'ed function, jax can check whether it's seen an array with the
  same static coordinates(*) before or whether it needs to recompile the
  function for the new values of the static coordinates.

  (*) note jax_coords are not included in this; their value can be different
  on different calls without triggering a recompile.
  """

  def __init__(self, coord_vars: Mapping[Hashable, xarray.Variable]):
    self._variables = coord_vars
    self._hash = None

  def __repr__(self) -> str:
    return f'_HashableCoords({repr(self._variables)})'

  def __getitem__(self, key: Hashable) -> xarray.Variable:
    return self._variables[key]

  def __len__(self) -> int:
    return len(self._variables)

  def __iter__(self) -> Iterator[Hashable]:
    return iter(self._variables)

  def __hash__(self):
    if self._hash is None:
      self._hash = hash(frozenset((name, var.data.tobytes())
                                  for name, var in self._variables.items()))
    return self._hash

  def __eq__(self, other):
    if self is other:
      return True
    elif not isinstance(other, type(self)):
      return NotImplemented
    elif self._variables is other._variables:
      return True
    else:
      return self._variables.keys() == other._variables.keys() and all(
          variable.equals(other._variables[name])
          for name, variable in self._variables.items())


def _flatten_data_array(v: xarray.DataArray) -> Tuple[
    # Children (data variable, jax_coord_vars):
    Tuple[xarray.Variable, Mapping[Hashable, xarray.Variable]],
    # Static auxiliary data (name, static_coord_vars):
    Tuple[Optional[Hashable], _HashableCoords]]:
  """Flattens a DataArray for jax.tree_util."""
  static_coord_vars, jax_coord_vars = _split_static_and_jax_coords(v.coords)
  children = (v.variable, jax_coord_vars)
  aux = (v.name, _HashableCoords(static_coord_vars))
  return children, aux


def _unflatten_data_array(
    aux: Tuple[Optional[Hashable], _HashableCoords],
    children: Tuple[xarray.Variable, Mapping[Hashable, xarray.Variable]],
) -> xarray.DataArray:
  """Unflattens a DataArray for jax.tree_util."""
  variable, jax_coord_vars = children
  name, static_coord_vars = aux
  if _DIMS_CHANGE_ON_UNFLATTEN_FN.get(None):
    # Drop static coords which have dims not present in any of the data_vars.
    # These would generally be dims that were dropped by a dims_change_fn, but
    # because static coordinates don't go through dims_change_fn on unflatten,
    # we just drop them where this causes a problem.
    # Since jax_coords go through the dims_change_fn on unflatten we don't need
    # to do this for jax_coords.
    static_coord_vars = _drop_with_none_of_dims(
        static_coord_vars, variable.dims)
  return core.DataArray(
      variable, name=name, coords=static_coord_vars, jax_coords=jax_coord_vars)


def _flatten_dataset(dataset: xarray.Dataset) -> Tuple[
    # Children (data variables, jax_coord_vars):
    Tuple[Mapping[Hashable, xarray.Variable],
          Mapping[Hashable, xarray.Variable]],
    # Static auxiliary data (static_coord_vars):
    _HashableCoords]:
  """Flattens a Dataset for jax.tree_util."""
  variables = {name: data_array.variable
               for name, data_array in dataset.data_vars.items()}
  static_coord_vars, jax_coord_vars = _split_static_and_jax_coords(
      dataset.coords)
  children = (variables, jax_coord_vars)
  aux = _HashableCoords(static_coord_vars)
  return children, aux


def _unflatten_dataset(
    aux: _HashableCoords,
    children: Tuple[Mapping[Hashable, xarray.Variable],
                    Mapping[Hashable, xarray.Variable]],
    ) -> xarray.Dataset:
  """Unflattens a Dataset for jax.tree_util."""
  data_vars, jax_coord_vars = children
  static_coord_vars = aux
  dataset = xarray.Dataset(data_vars)
  if _DIMS_CHANGE_ON_UNFLATTEN_FN.get(None):
    # Drop static coords which have dims not present in any of the data_vars.
    # See corresponding comment in _unflatten_data_array.
    static_coord_vars = _drop_with_none_of_dims(
        static_coord_vars, dataset.dims)  # pytype: disable=wrong-arg-types
  return core.assign_coords(
      dataset, coords=static_coord_vars, jax_coords=jax_coord_vars)


def _flatten_datatree(datatree: xarray.DataTree) -> Tuple[
    Tuple[Mapping[str, xarray.DataTree], xarray.Dataset], str | None]:
  """Flattens a DataTree for jax.tree_util."""
  # For simplicity we assume DataTrees will be flattened/unflattened from the
  # root. If you give it a non-root-node, it will still work but any parents
  # (and any coordinates inherited from them) will be lost.
  node_dataset = datatree.to_dataset(inherit=False)
  children = (dict(datatree.children), node_dataset)
  aux = datatree.name
  return children, aux


def _unflatten_datatree(
    aux: str | None,
    children: Tuple[Mapping[str, xarray.DataTree], xarray.Dataset],
) -> xarray.DataTree:
  """Unflattens a DataTree for jax.tree_util."""
  children_dict, node_dataset = children
  name = aux
  return xarray.DataTree(
      dataset=node_dataset, children=children_dict, name=name)


jax.tree_util.register_pytree_node(
    xarray.Variable, _flatten_variable, _unflatten_variable)
# This is a subclass of Variable but still needs registering separately.
# Flatten/unflatten for IndexVariable is a bit of a corner case but we do
# need to support it.
jax.tree_util.register_pytree_node(
    xarray.IndexVariable, _flatten_variable, _unflatten_variable)
jax.tree_util.register_pytree_node(
    xarray.DataArray, _flatten_data_array, _unflatten_data_array)
jax.tree_util.register_pytree_node(
    xarray.Dataset, _flatten_dataset, _unflatten_dataset)
jax.tree_util.register_pytree_node(
    xarray.DataTree, _flatten_datatree, _unflatten_datatree)
