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
"""Core user-facing constructors and coordinate handling functions."""

from typing import Any, Mapping, Optional, TypeVar
from typing import Hashable  # pylint: disable=deprecated-class

import jax
import jax.numpy as jnp
import numpy as np
import xarray

_JAX_COORD_ATTR_NAME = '_jax_coord'


def DataArray(  # pylint:disable=invalid-name
    data,
    coords=None,
    dims=None,
    name=None,
    attrs=None,
    jax_coords=None,
) -> xarray.DataArray:
  """Like xarray.DataArray, but supports using JAX arrays.

  Args:
    data: As for xarray.DataArray, except jax arrays are also supported.
    coords: Coordinates for the array, see xarray.DataArray. These coordinates
      must be based on plain numpy arrays or something convertible to plain
      numpy arrays. Their values will form a static part of the data structure
      from the point of view of jax.tree_util. In particular this means these
      coordinates will be passed as plain numpy arrays even inside a JIT'd
      function, and the JIT'd function will be recompiled under the hood if the
      coordinates of DataArrays passed into it change.
      If this is not convenient for you, see also jax_coords below.
    dims: See xarray.DataArray.
    name: See xarray.DataArray.
    attrs: See xarray.DataArray.
    jax_coords: Additional coordinates, which *can* use JAX arrays. These
      coordinates will be treated as JAX data from the point of view of
      jax.tree_util, that means when JIT'ing they will be passed as tracers and
      computation involving them will be JIT'd.
      Unfortunately a side-effect of this is that they can't be used as index
      coordinates (because xarray's indexing logic is not JIT-able). If you
      specify a coordinate with the same name as a dimension here, it will not
      be set as an index coordinate; this behaviour is different to the default
      for `coords`, and it means that things like `.sel` based on the jax
      coordinate will not work.
      Note we require `jax_coords` to be explicitly specified via a different
      constructor argument to `coords`, rather than just looking for jax arrays
      within the `coords` and treating them differently. This is because it
      affects the way jax.tree_util treats them, which is somewhat orthogonal to
      whether the value is passed in as numpy or not, and generally needs to be
      handled consistently so is something we encourage explicit control over.

  Returns:
    An instance of xarray.DataArray.
  """
  result = xarray.DataArray(data, dims=dims, name=name, attrs=attrs or {})
  return assign_coords(result, coords=coords, jax_coords=jax_coords)


def Dataset(  # pylint:disable=invalid-name
    data_vars=None,
    coords=None,
    attrs=None,
    jax_coords=None,
) -> xarray.Dataset:
  """Like xarray.Dataset, but can wrap JAX arrays.

  Args:
    data_vars: As for xarray.Dataset, except jax arrays are also supported.
    coords: Coordinates for the dataset, see xarray.Dataset. These coordinates
      must be based on plain numpy arrays or something convertible to plain
      numpy arrays. Their values will form a static part of the data structure
      from the point of view of jax.tree_util. In particular this means these
      coordinates will be passed as plain numpy arrays even inside a JIT'd
      function, and the JIT'd function will be recompiled under the hood if the
      coordinates of DataArrays passed into it change.
      If this is not convenient for you, see also jax_coords below.
    attrs: See xarray.Dataset.
    jax_coords: Additional coordinates, which *can* use JAX arrays. These
      coordinates will be treated as JAX data from the point of view of
      jax.tree_util, that means when JIT'ing they will be passed as tracers and
      computation involving them will be JIT'd.
      Unfortunately a side-effect of this is that they can't be used as index
      coordinates (because xarray's indexing logic is not JIT-able). If you
      specify a coordinate with the same name as a dimension here, it will not
      be set as an index coordinate; this behaviour is different to the default
      for `coords`, and it means that things like `.sel` based on the jax
      coordinate will not work.
      Note we require `jax_coords` to be explicitly specified via a different
      constructor argument to `coords`, rather than just looking for jax arrays
      within the `coords` and treating them differently. This is because it
      affects the way jax.tree_util treats them, which is somewhat orthogonal to
      whether the value is passed in as numpy or not, and generally needs to be
      handled consistently so is something we encourage explicit control over.

  Returns:
    An instance of xarray.Dataset.
  """
  result = xarray.Dataset(data_vars=data_vars, attrs=attrs)

  return assign_coords(result, coords=coords, jax_coords=jax_coords)


# Alias for backward-compatibility
Variable = xarray.Variable


DatasetOrDataArray = TypeVar(
    'DatasetOrDataArray', xarray.Dataset, xarray.DataArray)


def assign_coords(
    x: DatasetOrDataArray,
    *,
    coords: Optional[Mapping[Hashable, Any]] = None,
    jax_coords: Optional[Mapping[Hashable, Any]] = None,
) -> DatasetOrDataArray:
  """Replacement for assign_coords which works in presence of jax_coords.

  `jax_coords` allow certain specified coordinates to have their data passed as
  JAX arrays (including through jax.jit boundaries). The compromise in return is
  that they are not created as index coordinates and cannot be used for .sel
  and other coordinate-based indexing operations. See docs for `jax_coords` on
  xarray_jax.Dataset and xarray_jax.DataArray for more information.

  This function can be used to set jax_coords on an existing DataArray or
  Dataset, and also to set a mix of jax and non-jax coordinates. It uses
  xarray.Coordinates with empty indexes to prevent xarray trying and failing
  to create IndexVariables from jax arrays under the hood.

  Args:
    x: An xarray Dataset or DataArray.
    coords: Dict of (non-JAX) coords, or None if not assigning any.
    jax_coords: Dict of JAX coords, or None if not assigning any. See docs for
      xarray_jax.Dataset / DataArray for more information on jax_coords.

  Returns:
    The Dataset or DataArray with coordinates assigned, similarly to
    Dataset.assign_coords / DataArray.assign_coords.
  """
  jax_coords = {} if jax_coords is None else dict(jax_coords)

  # Assign static coordinates directly
  if coords: x = x.assign_coords(coords)

  # For each JAX coordinate, create a new xarray.Variable with empty indexes
  processed_jax_coords = {}
  for name, coord in jax_coords.items():
    if isinstance(coord, xarray.DataArray):
      coord = coord.variable

    if isinstance(coord, list):
      coord = np.array(coord)

    if isinstance(coord, xarray.Variable):
      coord = coord.copy(deep=False)  # Copy before mutating attrs.
    elif isinstance(coord, tuple):
      # A tuple represents a pair of (dims, data).
      dims, data = coord
      coord = xarray.Variable(dims, data)
    elif jnp.isscalar(coord):
      # A scalar coord maps to a scalar Variable:
      coord = xarray.Variable(dims=(), data=coord)
    elif isinstance(coord, jax.typing.ArrayLike) and jnp.ndim(coord) == 1:
      # A 1D array maps to a 1D Variable whose dimension is the same as the
      # coordinate name:
      coord = xarray.Variable((name,), coord)
    else:
      raise ValueError(f'Unsupported value for coordinate {name}')

    # We set an attr on each jax_coord identifying it as such. These attrs on
    # the coord Variable gets reflected on the coord DataArray exposed too, and
    # when set on coordinates they generally get preserved under the default
    # keep_attrs setting.
    # These attrs are used by jax.tree_util registered flatten/unflatten to
    # determine which coords need to be treated as leaves of the flattened
    # structure vs static data.
    coord.attrs[_JAX_COORD_ATTR_NAME] = True
    processed_jax_coords[name] = coord

  # Use xarray.Coordinates with empty indexes to skip automatic index creation
  jax_coords_obj = xarray.Coordinates(coords=processed_jax_coords, indexes={})
  x = x.assign_coords(jax_coords_obj)

  return x


def get_jax_coords(x: DatasetOrDataArray) -> Mapping[Hashable, Any]:
  return {
      name: coord_var
      for name, coord_var in x.coords.variables.items()
      if coord_var.attrs.get(_JAX_COORD_ATTR_NAME, False)}


def assign_jax_coords(
    x: DatasetOrDataArray,
    jax_coords: Optional[Mapping[Hashable, Any]] = None,
    **jax_coords_kwargs
) -> DatasetOrDataArray:
  """Assigns only jax_coords, with same API as xarray's assign_coords."""
  return assign_coords(x, jax_coords=jax_coords or jax_coords_kwargs)

# Alias for backward-compatibility
apply_ufunc = xarray.apply_ufunc
