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
"""Helpers to use xarray.{Variable,DataArray,Dataset} with JAX.

Allows them to be used within JAX transformations, so you can start with a JAX
array, do some computation with it in xarray-land, get a JAX array out the other
end, and (for example) `jax.jit` through the whole thing. You can even `jax.jit`
a function which accepts and returns xarray.Dataset, DataArray and Variable.

## Custom Constructors and Unwrapping

While xarray's standard constructors now support JAX arrays directly, this
library also provides custom xarray_jax.{DataArray, Dataset} constructors to
support `jax_coords` (see "Treatment of coordinates" below).

To get non-wrapped JAX arrays out the other end, you can use e.g.:

  xarray_jax.jax_vars(dataset)
  xarray_jax.jax_data(dataset.some_var)

which will complain if the data isn't actually a JAX array. Use this if you need
to make sure the computation has gone via JAX, e.g. if it's the output of code
that you want to JIT or compute gradients through. If this is not the case and
you want to support passing plain numpy arrays through as well as potentially
JAX arrays, you can use:

  xarray_jax.unwrap_vars(dataset)
  xarray_jax.unwrap_data(dataset.some_var)

which will return the underlying data.

## jax.jit and pmap of functions taking and returning xarray datatypes

We register xarray datatypes with jax.tree_util, which allows them to be treated
as generic containers of jax arrays by various parts of jax including jax.jit.

This allows for, e.g.:

  @jax.jit
  def foo(input: xarray.Dataset) -> xarray.Dataset:
    ...

It will not work out-of-the-box with shape-modifying transformations like
jax.pmap, or e.g. a jax.tree_util.tree_map with some transform that alters array
shapes or dimension order. That's because we won't know what dimension names
and/or coordinates to use when unflattening, if the results have a different
shape to the data that was originally flattened.

You can work around this using xarray_jax.dims_change_on_unflatten, however,
and in the case of jax.pmap we provide a wrapper xarray_jax.pmap which allows
it to be used with functions taking and returning xarrays.

## Treatment of coordinates

We don't support passing jax arrays as coordinates when constructing a
DataArray/Dataset. This is because xarray's advanced indexing and slicing is
unlikely to work with jax arrays (at least when a Tracer is used during
jax.jit), and also because some important datatypes used for coordinates, like
timedelta64 and datetime64, are not supported by jax.

For the purposes of tree_util and jax.jit, coordinates are not treated as leaves
of the tree (array data 'contained' by a Dataset/DataArray), they are just a
static part of the structure. That means that if a jit'ed function is called
twice with Dataset inputs that use different coordinates, it will compile a
separate version of the function for each. The coordinates are treated like
static_argnums by jax.jit.

If you want to use dynamic data for coordinates, we recommend making it a
data_var instead of a coord. You won't be able to do indexing and slicing using
the coordinate, but that wasn't going to work with a jax array anyway.
"""

__version__ = "0.0.1"

# pylint: disable=g-importing-member,g-multiple-import

from xarray_jax.core import (
    apply_ufunc,
    DataArray,
    Dataset,
    assign_coords,
    assign_jax_coords,
    get_jax_coords,
    Variable,
)
from xarray_jax.jax_transforms import (
    vmap,
    pmap,
    scan,
    tree_map_variables,
    tree_map_with_dims,
)
from xarray_jax.pytree import (
    dims_change_on_unflatten,
    jax_data,
    jax_vars,
    NonArrayLeafWrapper,
    unwrap,
    unwrap_coords,
    unwrap_data,
    unwrap_vars,
    wrap,
)

__all__ = (
    # core
    'apply_ufunc',
    'DataArray',
    'Dataset',
    'assign_coords',
    'assign_jax_coords',
    'get_jax_coords',
    'Variable',
    # jax_transforms
    'vmap',
    'pmap',
    'scan',
    'tree_map_variables',
    'tree_map_with_dims',
    # pytree
    'dims_change_on_unflatten',
    'jax_data',
    'jax_vars',
    'NonArrayLeafWrapper',
    'unwrap',
    'unwrap_coords',
    'unwrap_data',
    'unwrap_vars',
    'wrap',
)