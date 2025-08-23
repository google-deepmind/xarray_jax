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
"""Wrappers around JAX transformations."""

from typing import Any, Callable, Optional, TypeVar

import jax
import xarray

from xarray_jax.pytree import dims_change_on_unflatten, unwrap



def pmap(fn: Callable[..., Any],
         dim: str,
         axis_name: Optional[str] = None,
         devices: ... = None,
         backend: ... = None) -> Callable[..., Any]:
  """Wraps a subset of jax.pmap functionality to handle xarray input/output.

  Constraints:
    * Any Dataset or DataArray passed to the function must have `dim` as the
      first dimension. This will be checked. You can ensure this if necessary
      by calling `.transpose(dim, ...)` beforehand.
    * All args and return values will be mapped over the first dimension,
      it will use in_axes=0, out_axes=0.
    * No support for static_broadcasted_argnums, donate_argnums etc.

  Args:
    fn: Function to be pmap'd which takes and returns trees which may contain
      xarray Dataset/DataArray. Any Dataset/DataArrays passed as input must use
      `dim` as the first dimension on all arrays.
    dim: The xarray dimension name corresponding to the first dimension that is
      pmapped over (pmap is called with in_axes=0, out_axes=0).
    axis_name: Used by jax to identify the mapped axis so that parallel
      collectives can be applied. Defaults to same as `dim`.
    devices:
    backend:
      See jax.pmap.

  Returns:
    A pmap'd version of `fn`, which takes and returns Dataset/DataArray with an
    extra leading dimension `dim` relative to what the original `fn` sees.
  """
  input_treedef = None
  output_treedef = None

  def fn_passed_to_pmap(*flat_args):
    assert input_treedef is not None
    # Inside the pmap the original first dimension will no longer be present:
    def check_and_remove_leading_dim(dims):
      try:
        index = dims.index(dim)
      except ValueError:
        index = None
      if index != 0:
        raise ValueError(f'Expected dim {dim} at index 0, found at {index}.')
      return dims[1:]
    with dims_change_on_unflatten(check_and_remove_leading_dim):
      args = jax.tree_util.tree_unflatten(input_treedef, flat_args)
    result = fn(*args)
    nonlocal output_treedef
    flat_result, output_treedef = jax.tree_util.tree_flatten(result)
    return flat_result

  pmapped_fn = jax.pmap(
      fn_passed_to_pmap,
      axis_name=axis_name or dim,
      in_axes=0,
      out_axes=0,
      devices=devices,
      backend=backend)

  def result_fn(*args):
    nonlocal input_treedef
    flat_args, input_treedef = jax.tree_util.tree_flatten(args)
    flat_result = pmapped_fn(*flat_args)
    assert output_treedef is not None
    # After the pmap an extra leading axis will be present, we need to add an
    # xarray dimension for this when unflattening the result:
    with dims_change_on_unflatten(lambda dims: (dim,) + dims):
      return jax.tree_util.tree_unflatten(output_treedef, flat_result)

  return result_fn

_PyTree = TypeVar('_PyTree')


def tree_map_variables(
    func: Callable[[xarray.Variable], xarray.Variable],
    tree_data: _PyTree) -> _PyTree:
  """Like jax.tree.map but operates with Variables as leaves.

  This will work with any jax.tree_util-registered PyTree containing xarray
  datatypes. All jax data in xarray datatypes is exposed via xarray.Variable
  nodes by our registered flatten/unflatten functions and hence here too. Note
  static coordinate data will not be mapped over however.

  This allows you to see the associated dimensions for each leaf, and to change
  them. If you change them, it's your responsibility to ensure that when
  unflattened back into DataArray/Dataset/DataTree the result still makes sense.
  In particular that any updated shapes are consistent with the shapes of any
  static (non-jax_coord) coordinates, since these will not be mapped over.

  Args:
    func: Function from xarray.Variable to xarray.Variable.
    tree_data: PyTree to be mapped over.

  Returns:
    PyTree with the same structure as `tree_data` but where xarray.Variables
    within xarray datatypes have been mapped over by `func`. Any leaves outside
    of xarray datatypes will be unchanged.
  """
  return jax.tree.map(
      lambda leaf: func(leaf) if isinstance(leaf, xarray.Variable) else leaf,
      tree_data,
      is_leaf=lambda x: isinstance(x, xarray.Variable))


def tree_map_with_dims(
    func: Callable[[jax.typing.ArrayLike, tuple[str, ...] | None],
                   jax.typing.ArrayLike],
    data: _PyTree,
) -> _PyTree:
  """Like jax.tree.map but also passes in xarray dimensions where known.

  This is convenient when applying logic to every jax array in some xarray data
  structure, which wants to be sensitive to the xarray dimension names.
  Typical examples of this would be jax operations relating to sharding where
  you may want to map xarray dimension names to sharding axis names.

  This only supports changing array shapes in limited situations (see below).

  Unlike tree_map_variables above, this will also map over plain jax arrays
  that don't occur within xarray.Variable nodes; these will be passed to func
  with dims=None.

  Args:
    func: A function from (jax_array, dims) -> jax_array. dims will correspond
      to the dimension names of the xarray.Variable containing the jax_array
      where it occurs within an xarray.Variable, note this includes arrays
      within xarray.Dataset and xarray.DataArray too. For plain jax arrays that
      don't occur within an xarray.Variable, dims will be None.
      The returned jax array should generally be of the same shape as the input.
      However you can get away with changing the shape of a particular dimension
      in limited circumstances: when there are no explicit coordinates involving
      that dimension, or when the only coordinates involving that dimension are
      jax_coords and you modify their shapes too in a consistent fashion.
      You are not allowed to change the dimension order, add or remove
      dimensions.
    data: Any pytree with jax ArrayLike leaves suitable for use with
      `jax.tree.map`. Thanks to xarray_jax such pytrees may include xarray
      datatypes.

  Returns:
    A pytree of the same structure as data, with the result of applying func
    to each jax array found.
  """
  # All jax arrays within xarray.Dataset, xarray.DataArray (including
  # jax_coord arrays) will be exposed via xarray.Variable internal nodes by
  # xarray_jax's pytree registrations. So to find xarray dimension metadata
  # it's sufficient to stop descending at xarray.Variable nodes:
  def is_leaf(x):
    return isinstance(x, xarray.Variable)

  def wrapped_func(x):
    if isinstance(x, xarray.Variable):
      array = unwrap(x.data)
      array = func(array, x.dims)
      return xarray.Variable(dims=x.dims, data=array)
    else:
      return func(x, None)

  return jax.tree_util.tree_map(wrapped_func, data, is_leaf=is_leaf)


_Carry = TypeVar('_Carry')
_X = TypeVar('_X')
_Y = TypeVar('_Y')


def scan(f: Callable[[_Carry, _X], tuple[_Carry, _Y]],
         init: _Carry,
         dim: str,
         xs: _X | None = None,
         length: int | None = None,
         reverse: bool = False,
         unroll: int | bool = 1,
         ) -> tuple[_Carry, _Y]:
  """Like jax.lax.scan but supports xarray data.

  This can handle a jax.tree containing any mix of xarray and plain jax data.
  It scans along the dimension `dim` for xarray data, and the leading axis for
  any non-xarray data. These scanned-along dimensions must all be consistent in
  size.

  Static coordinates along `dim` in the `xs` will not be present on the `x`
  argument to `f`, since they would be different on each iteration and we can't
  pass them through as static data. jax_coords will be passed through correctly
  however.

  Static coordinates along `dim` on the `xs` will also not be present on the
  resulting `ys`, you will need to copy these across yourself if desired.
  (This one we may be able to fix in future.)

  Args:
    f: Function to apply at each step of the scan. This should map
      (carry, x) -> (carry, y), where `x` is a slice of the `xs` along the `dim`
      axis (with the `dim` axis and any coordinates using it dropped), and `y`
      is a slice of the desired output along the `dim` axis, with no `dim` axis
      itself.
      x, y and carry can in general be trees, in which the above applies to
      each leaf of the tree.
    init: Initial value of the carry.
    dim: The xarray dimension name to scan along, for xarray data.
    xs: The input to be scanned. If not provided, will scan over `length`
      iterations with None passed as the `x` argument to `f`.
    length: The length of the scan, if `xs` are not provided.
    reverse: Whether to scan in reverse order.
    unroll: How many steps to unroll the scan, see jax.lax.scan.

  Returns:
    final_carry: The carry returned from the final step of the scan.
    ys: Data corresponding to the `y` returned from `f` on each step of the
      scan, concatenated along an extra leading dimension (named `dim` for
      xarray data).
  """
  if xs is not None:
    # Ensure `dim` is the leading axis on any xarray data in the `xs`. This is
    # what jax.lax.scan will scan over. (Any non-array data it's on you to put
    # the relevant axis first).
    xs = tree_map_variables(lambda v: v.transpose(dim, ...), xs)
    xs_leaves, xs_treedef = jax.tree.flatten(xs)
  else:
    xs_treedef = None
    xs_leaves = None

  y_treedef = None

  def scan_fn(carry, x_leaves):
    if x_leaves is None:
      x = None
    else:
      with dims_change_on_unflatten(lambda dims: dims[1:]):
        x = jax.tree.unflatten(xs_treedef, x_leaves)
    carry, y = f(carry, x)

    nonlocal y_treedef
    y_leaves, y_treedef = jax.tree.flatten(y)
    return carry, y_leaves

  final_carry, ys_leaves = jax.lax.scan(
      scan_fn,
      init,
      xs_leaves,
      length=length,
      reverse=reverse,
      unroll=unroll)

  assert isinstance(y_treedef, jax.tree_util.PyTreeDef)

  with dims_change_on_unflatten(lambda dims: (dim,) + dims):
    ys = jax.tree.unflatten(y_treedef, ys_leaves)

  return final_carry, ys
