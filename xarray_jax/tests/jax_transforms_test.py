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


class JaxTransformsTest(absltest.TestCase):

  def test_pmap(self):
    devices = jax.local_device_count()
    foo = jnp.zeros((devices, 3, 4), dtype=np.float32)
    bar = jnp.zeros((devices, 2, 3, 4), dtype=np.float32)
    dataset = xarray_jax.Dataset({
        'foo': (('device', 'lat', 'lon'), foo),
        'bar': (('device', 'time', 'lat', 'lon'), bar)})

    def func(d):
      self.assertNotIn('device', d.dims)
      return d + 1
    func = xarray_jax.pmap(func, dim='device')

    result = func(dataset)
    xarray.testing.assert_identical(
        jax.device_get(dataset + 1),
        jax.device_get(result))

    # Can call it again with a different argument structure (it will recompile
    # under the hood but should work):
    dataset = dataset.drop_vars('foo')
    result = func(dataset)
    xarray.testing.assert_identical(
        jax.device_get(dataset + 1),
        jax.device_get(result))

  def test_pmap_with_jax_coords(self):
    devices = jax.local_device_count()
    foo = jnp.zeros((devices, 3, 4), dtype=np.float32)
    bar = jnp.zeros((devices, 2, 3, 4), dtype=np.float32)
    time = jnp.zeros((devices, 2), dtype=np.float32)
    dataset = xarray_jax.Dataset(
        {'foo': (('device', 'lat', 'lon'), foo),
         'bar': (('device', 'time', 'lat', 'lon'), bar)},
        coords={
            'lat': np.arange(3),
            'lon': np.arange(4),
        },
        jax_coords={
            # Currently any jax_coords need a leading device dimension to use
            # with pmap, same as for data_vars.
            # TODO(matthjw): have pmap automatically broadcast to all devices
            # where the device dimension not present.
            'time': xarray.Variable(('device', 'time'), time),
        }
    )

    def func(d):
      self.assertNotIn('device', d.dims)
      self.assertNotIn('device', d.coords['time'].dims)

      # The jax_coord 'time' should be passed in backed by a JAX array, but
      # not as an index coordinate.
      self.assertIsInstance(d.coords['time'].data, jax.Array)
      self.assertNotIn('time', d.indexes)

      return d + 1
    func = xarray_jax.pmap(func, dim='device')

    result = func(dataset)
    xarray.testing.assert_identical(
        jax.device_get(dataset + 1),
        jax.device_get(result))

    # Can call it again with a different argument structure (it will recompile
    # under the hood but should work):
    dataset = dataset.drop_vars('foo')
    result = func(dataset)
    xarray.testing.assert_identical(
        jax.device_get(dataset + 1),
        jax.device_get(result))

  def test_pmap_with_tree_mix_of_xarray_and_jax_array(self):
    devices = jax.local_device_count()
    data_array = xarray_jax.DataArray(
        data=jnp.ones((devices, 3, 4), dtype=np.float32),
        dims=('device', 'lat', 'lon'))
    plain_array = jnp.ones((devices, 2), dtype=np.float32)
    inputs = {'foo': data_array,
              'bar': plain_array}

    def func(x):
      return x['foo'] + 1, x['bar'] + 1

    func = xarray_jax.pmap(func, dim='device')
    result_foo, result_bar = func(inputs)
    xarray.testing.assert_identical(
        jax.device_get(inputs['foo'] + 1),
        jax.device_get(result_foo))
    np.testing.assert_array_equal(
        jax.device_get(inputs['bar'] + 1),
        jax.device_get(result_bar))

  def test_pmap_complains_when_dim_not_first(self):
    devices = jax.local_device_count()
    data_array = xarray_jax.DataArray(
        data=jnp.ones((3, devices, 4), dtype=np.float32),
        dims=('lat', 'device', 'lon'))

    func = xarray_jax.pmap(lambda x: x+1, dim='device')
    with self.assertRaisesRegex(
        ValueError, 'Expected dim device at index 0, found at 1'):
      func(data_array)

  def test_apply_ufunc(self):
    inputs = xarray_jax.DataArray(
        data=jnp.asarray([[1, 2], [3, 4]]),
        dims=('x', 'y'),
        coords={'x': [0, 1],
                'y': [2, 3]})
    result = xarray_jax.apply_ufunc(
        lambda x: jnp.sum(x, axis=-1),
        inputs,
        input_core_dims=[['x']])
    expected_result = xarray_jax.DataArray(
        data=[4, 6],
        dims=('y',),
        coords={'y': [2, 3]})
    xarray.testing.assert_identical(expected_result, jax.device_get(result))

  def test_apply_ufunc_multiple_return_values(self):
    def ufunc(array):
      return jnp.min(array, axis=-1), jnp.max(array, axis=-1)
    inputs = xarray_jax.DataArray(
        data=jnp.asarray([[1, 4], [3, 2]]),
        dims=('x', 'y'),
        coords={'x': [0, 1],
                'y': [2, 3]})
    result = xarray_jax.apply_ufunc(
        ufunc, inputs, input_core_dims=[['x']], output_core_dims=[[], []])
    expected = (
        # Mins:
        xarray_jax.DataArray(
            data=[1, 2],
            dims=('y',),
            coords={'y': [2, 3]}
        ),
        # Maxes:
        xarray_jax.DataArray(
            data=[3, 4],
            dims=('y',),
            coords={'y': [2, 3]}
        )
    )
    xarray.testing.assert_identical(expected[0], jax.device_get(result[0]))
    xarray.testing.assert_identical(expected[1], jax.device_get(result[1]))

  def test_scan(self):
    def f(carry, x):
      dataset_input, plain_jax_input = x
      self.assertEqual(plain_jax_input.shape, ())
      self.assertEqual(dataset_input.sizes, {'x': 2})  # No 'time' dimension.
      carry = carry + 1
      y = dataset_input + carry
      return carry, y

    dataset_inputs = xarray_jax.Dataset(
        data_vars={
            # Put the scan dimension (time) second to make sure it transposes
            # the array appropriately for JAX which requires scan dim first.
            'foo': (('x', 'time'), jnp.zeros((2, 5))),
        },
        coords={
            'x': np.arange(2),
            'time': np.arange(5) * 10,
        },
        jax_coords={
            'time_extra': (('time',), np.arange(5)*2),
        })
    # These must have scan dimension first:
    plain_jax_inputs = jnp.zeros((5,))

    carry, result = xarray_jax.scan(f,
                                    init=0,
                                    xs=(dataset_inputs, plain_jax_inputs),
                                    dim='time')
    self.assertEqual(jax.device_get(carry), 5)
    self.assertEqual(result.foo.sizes, {'x': 2, 'time': 5})
    self.assertIn('x', result.coords)
    # Unfortunately static coordinates along the dimension we scan over will
    # not be preserved on the result:
    self.assertNotIn('time', result.coords)
    # The jax_coord along time will still be present though:
    np.testing.assert_array_equal(
        jax.device_get(result.foo.coords['time_extra'].data),
        jax.device_get(dataset_inputs.coords['time_extra'].data))

  def test_scan_no_inputs(self):
    def f(carry, x):
      assert x is None
      carry = carry + 1
      y = carry * 10
      # Set a jax_coord on the output to check it maps through to the result
      # correctly:
      y = xarray_jax.assign_jax_coords(y, extra_coord=carry[0])
      return carry, y

    init = xarray_jax.DataArray(data=np.zeros(2), dims=('x',))
    carry, result = xarray_jax.scan(f, init=init, length=5, dim='time')
    np.testing.assert_array_equal(jax.device_get(carry), [5, 5])
    self.assertEqual(result.sizes, {'x': 2, 'time': 5})
    np.testing.assert_array_equal(jax.device_get(result.extra_coord.data),
                                  [1, 2, 3, 4, 5])


if __name__ == '__main__':
  absltest.main()