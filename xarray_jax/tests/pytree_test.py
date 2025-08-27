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

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import xarray
import xarray_jax
import xarray_jax.pytree


class PytreeTest(absltest.TestCase):

  def test_flatten_unflatten_variable(self):
    variable = xarray.Variable(
        ('lat', 'lon'), jnp.ones((3, 4), dtype=np.float32))
    children, aux = xarray_jax.pytree._flatten_variable(variable)
    # Check auxiliary info is hashable/comparable (important for jax.jit):
    hash(aux)
    self.assertEqual(aux, aux)
    roundtrip = xarray_jax.pytree._unflatten_variable(aux, children)
    self.assertTrue(variable.equals(roundtrip))

  def test_flatten_unflatten_data_array(self):
    data_array = xarray_jax.DataArray(
        data=jnp.ones((3, 4), dtype=np.float32),
        dims=('lat', 'lon'),
        coords={'lat': np.arange(3)},
        jax_coords={'lon': np.arange(4) * 10},
    )
    children, aux = xarray_jax.pytree._flatten_data_array(data_array)
    # Check auxiliary info is hashable/comparable (important for jax.jit):
    hash(aux)
    self.assertEqual(aux, aux)
    roundtrip = xarray_jax.pytree._unflatten_data_array(aux, children)
    self.assertTrue(data_array.equals(roundtrip))

  def test_flatten_unflatten_dataset(self):
    foo = jnp.ones((3, 4), dtype=np.float32)
    bar = jnp.ones((2, 3, 4), dtype=np.float32)
    dataset = xarray_jax.Dataset(
        data_vars={'foo': (('lat', 'lon'), foo),
                   'bar': (('time', 'lat', 'lon'), bar)},
        coords={
            'time': np.arange(2),
            'lat': np.arange(3) * 10},
        jax_coords={
            'lon': np.arange(4) * 10})
    children, aux = xarray_jax.pytree._flatten_dataset(dataset)
    # Check auxiliary info is hashable/comparable (important for jax.jit):
    hash(aux)
    self.assertEqual(aux, aux)
    roundtrip = xarray_jax.pytree._unflatten_dataset(aux, children)
    self.assertTrue(dataset.equals(roundtrip))

  def test_flatten_unflatten_datatree(self):
    # Coords to be inherited from the parent dataset, we include one jax
    # coord and one not to check both code paths
    parent_dataset = xarray_jax.Dataset(
        jax_coords={'time': xarray.Variable(('time',), np.arange(2))},
        coords={'lon': xarray.Variable(('lon',), np.arange(4) * 10)})

    bar = jnp.ones((2, 3, 4), dtype=np.float32)
    child_dataset = xarray_jax.Dataset(
        {'bar': (('time', 'lat', 'lon'), bar)},
        coords={'lat': np.arange(3)})

    datatree = xarray.DataTree(
        dataset=parent_dataset,
        children={'child': xarray.DataTree(dataset=child_dataset)})

    children, aux = xarray_jax.pytree._flatten_datatree(datatree)
    # Check auxiliary info is hashable/comparable (important for jax.jit):
    hash(aux)
    self.assertEqual(aux, aux)
    roundtrip = xarray_jax.pytree._unflatten_datatree(aux, children)
    self.assertTrue(datatree.equals(roundtrip))

  def test_flatten_unflatten_added_dim(self):
    data_array = xarray_jax.DataArray(
        data=jnp.ones((3, 4), dtype=np.float32),
        dims=('lat', 'lon'),
        coords={'lat': np.arange(3),
                'lon': np.arange(4) * 10})
    leaves, treedef = jax.tree_util.tree_flatten(data_array)
    leaves = [jnp.expand_dims(x, 0) for x in leaves]
    with xarray_jax.dims_change_on_unflatten(lambda dims: ('new',) + dims):
      with_new_dim = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertEqual(('new', 'lat', 'lon'), with_new_dim.dims)
    xarray.testing.assert_identical(
        jax.device_get(data_array),
        jax.device_get(with_new_dim.isel(new=0)))

  def test_map_added_dim(self):
    data_array = xarray_jax.DataArray(
        data=jnp.ones((3, 4), dtype=np.float32),
        dims=('lat', 'lon'),
        coords={'lat': np.arange(3),
                'lon': np.arange(4) * 10})
    with xarray_jax.dims_change_on_unflatten(lambda dims: ('new',) + dims):
      with_new_dim = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0),
                                            data_array)
    self.assertEqual(('new', 'lat', 'lon'), with_new_dim.dims)
    xarray.testing.assert_identical(
        jax.device_get(data_array),
        jax.device_get(with_new_dim.isel(new=0)))

  def test_map_remove_dim(self):
    foo = jnp.ones((1, 3, 4), dtype=np.float32)
    bar = jnp.ones((1, 2, 3, 4), dtype=np.float32)
    dataset = xarray_jax.Dataset(
        data_vars={'foo': (('batch', 'lat', 'lon'), foo),
                   'bar': (('batch', 'time', 'lat', 'lon'), bar)},
        coords={
            'batch': np.array([123]),
            'time': np.arange(2),
            'lat': np.arange(3) * 10,
            'lon': np.arange(4) * 10})
    with xarray_jax.dims_change_on_unflatten(lambda dims: dims[1:]):
      with_removed_dim = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, 0),
                                                dataset)
    self.assertEqual(('lat', 'lon'), with_removed_dim['foo'].dims)
    self.assertEqual(('time', 'lat', 'lon'), with_removed_dim['bar'].dims)
    self.assertNotIn('batch', with_removed_dim.dims)
    self.assertNotIn('batch', with_removed_dim.coords)
    xarray.testing.assert_identical(
        jax.device_get(dataset.isel(batch=0, drop=True)),
        jax.device_get(with_removed_dim))


if __name__ == '__main__':
  absltest.main()
