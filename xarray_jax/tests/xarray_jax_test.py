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
import chex
import jax
import jax.numpy as jnp
import numpy as np
import xarray
import xarray.ufuncs as xu
import xarray_jax


class XarrayJaxTest(absltest.TestCase):

  def test_jax_xarray_variable(self):
    def ops_via_xarray(inputs):
      x = xarray.Variable(('lat', 'lon'), inputs)
      # We'll apply a sequence of operations just to test that the end result is
      # still a JAX array, i.e. we haven't converted to numpy at any point.
      x = xu.abs((x + 2) * (x - 3))
      x = x.isel({'lat': slice(0, -1), 'lon': slice(1, 3)})
      x = xarray.Variable.concat([x, x + 1], dim='lat')
      x = x.transpose('lon', 'lat')
      x = x.stack(channels=('lon', 'lat'))
      x = x.sum()
      return xarray_jax.jax_data(x)

    # Check it doesn't leave jax-land when passed concrete values:
    ones = jnp.ones((3, 4), dtype=np.float32)
    result = ops_via_xarray(ones)
    self.assertIsInstance(result, jax.Array)

    # And that you can JIT it and compute gradients through it. These will
    # involve passing jax tracers through the xarray computation:
    jax.jit(ops_via_xarray)(ones)
    jax.grad(ops_via_xarray)(ones)

  def test_jax_xarray_data_array(self):
    def ops_via_xarray(inputs):
      x = xarray_jax.DataArray(dims=('lat', 'lon'),
                               data=inputs,
                               coords={'lat': np.arange(3) * 10,
                                       'lon': np.arange(4) * 10})
      x = xu.abs((x + 2) * (x - 3))
      x = x.sel({'lat': slice(0, 20)})
      y = xarray_jax.DataArray(dims=('lat', 'lon'),
                               data=ones,
                               coords={'lat': np.arange(3, 6) * 10,
                                       'lon': np.arange(4) * 10})
      x = xarray.concat([x, y], dim='lat')
      x = x.transpose('lon', 'lat')
      x = x.stack(channels=('lon', 'lat'))
      x = x.unstack()
      x = x.sum()
      return xarray_jax.jax_data(x)

    ones = jnp.ones((3, 4), dtype=np.float32)
    result = ops_via_xarray(ones)
    self.assertIsInstance(result, jax.Array)

    jax.jit(ops_via_xarray)(ones)
    jax.grad(ops_via_xarray)(ones)

  def test_jax_xarray_dataset(self):
    def ops_via_xarray(foo, bar):
      x = xarray_jax.Dataset(
          data_vars={'foo': (('lat', 'lon'), foo),
                     'bar': (('time', 'lat', 'lon'), bar)},
          coords={
              'time': np.arange(2),
              'lat': np.arange(3) * 10,
              'lon': np.arange(4) * 10})
      x = xu.abs((x + 2) * (x - 3))
      x = x.sel({'lat': slice(0, 20)})
      y = xarray_jax.Dataset(
          data_vars={'foo': (('lat', 'lon'), foo),
                     'bar': (('time', 'lat', 'lon'), bar)},
          coords={
              'time': np.arange(2),
              'lat': np.arange(3, 6) * 10,
              'lon': np.arange(4) * 10})
      x = xarray.concat([x, y], dim='lat')
      x = x.transpose('lon', 'lat', 'time')
      x = x.stack(channels=('lon', 'lat'))
      x = (x.foo + x.bar).sum()
      return xarray_jax.jax_data(x)

    foo = jnp.ones((3, 4), dtype=np.float32)
    bar = jnp.ones((2, 3, 4), dtype=np.float32)
    result = ops_via_xarray(foo, bar)
    self.assertIsInstance(result, jax.Array)

    jax.jit(ops_via_xarray)(foo, bar)
    jax.grad(ops_via_xarray)(foo, bar)

  def test_jit_function_with_xarray_variable_arguments_and_return(self):
    function = jax.jit(lambda v: v + 1)
    with self.subTest('jax input'):
      inputs = xarray.Variable(
          ('lat', 'lon'), jnp.ones((3, 4), dtype=np.float32))
      _ = function(inputs)
      # We test running the jitted function a second time, to exercise logic in
      # jax which checks if the structure of the inputs (including dimension
      # names and coordinates) is the same as it was for the previous call and
      # so whether it needs to re-trace-and-compile a new version of the
      # function or not. This can run into problems if the 'aux' structure
      # returned by the registered flatten function is not hashable/comparable.
      outputs = function(inputs)
      self.assertEqual(outputs.dims, inputs.dims)
    with self.subTest('numpy input'):
      inputs = xarray.Variable(
          ('lat', 'lon'), np.ones((3, 4), dtype=np.float32))
      _ = function(inputs)
      outputs = function(inputs)
      self.assertEqual(outputs.dims, inputs.dims)

  def test_jit_ahead_of_time_compile_with_xarray(self):
    # This needs jax.stages.ArgInfo to be wrapped, since .lower maps the xarray
    # to a pytree of ArgInfo under the hood.
    function = jax.jit(lambda v: v + 1)
    inputs = xarray.Variable(
        ('lat', 'lon'), jnp.ones((3, 4), dtype=np.float32))
    compiled_function = function.lower(inputs).compile()
    outputs = compiled_function(inputs)
    self.assertEqual(outputs.dims, inputs.dims)

  def test_jit_problem_if_convert_to_plain_numpy_array(self):
    inputs = xarray_jax.DataArray(
        data=jnp.ones((2,), dtype=np.float32), dims=('foo',))
    with self.assertRaises(jax.errors.TracerArrayConversionError):
      # Calling .values on a DataArray converts its values to numpy:
      jax.jit(lambda data_array: data_array.values)(inputs)

  def test_grad_function_with_xarray_variable_arguments(self):
    x = xarray.Variable(('lat', 'lon'), jnp.ones((3, 4), dtype=np.float32))
    # For grad we still need a JAX scalar as the output:
    jax.grad(lambda v: xarray_jax.jax_data(v.sum()))(x)

  def test_jit_function_with_xarray_data_array_arguments_and_return(self):
    inputs = xarray_jax.DataArray(
        data=jnp.ones((3, 4), dtype=np.float32),
        dims=('lat', 'lon'),
        coords={'lat': np.arange(3),
                'lon': np.arange(4) * 10})
    fn = jax.jit(lambda v: v + 1)
    _ = fn(inputs)
    outputs = fn(inputs)
    self.assertEqual(outputs.dims, inputs.dims)
    chex.assert_trees_all_equal(outputs.coords, inputs.coords)

  def test_jit_function_with_data_array_and_jax_coords(self):
    inputs = xarray_jax.DataArray(
        data=jnp.ones((3, 4), dtype=np.float32),
        dims=('lat', 'lon'),
        coords={'lat': np.arange(3)},
        jax_coords={'lon': jnp.arange(4) * 10})
    # Verify the jax_coord 'lon' retains jax data, and has not been created
    # as an index coordinate:
    self.assertIsInstance(inputs.coords['lon'].data, jax.Array)
    self.assertNotIn('lon', inputs.indexes)

    @jax.jit
    def fn(v):
      # The non-JAX coord is passed with numpy array data and an index:
      self.assertIsInstance(v.coords['lat'].data, np.ndarray)
      self.assertIn('lat', v.indexes)

      # The jax_coord is passed with JAX array data:
      self.assertIsInstance(v.coords['lon'].data, jax.Array)
      self.assertNotIn('lon', v.indexes)

      # Use the jax coord in the computation:
      v = v + v.coords['lon']

      # Return with an updated jax coord:
      return xarray_jax.assign_jax_coords(v, lon=v.coords['lon'] + 1)

    _ = fn(inputs)
    outputs = fn(inputs)

    # Verify the jax_coord 'lon' has jax data in the output too:
    self.assertIsInstance(
        outputs.coords['lon'].data, jax.Array)
    self.assertNotIn('lon', outputs.indexes)

    self.assertEqual(outputs.dims, inputs.dims)
    chex.assert_trees_all_equal(outputs.coords['lat'], inputs.coords['lat'])
    # Check our computations with the coordinate values worked:
    chex.assert_trees_all_equal(
        outputs.coords['lon'].data, (inputs.coords['lon']+1).data)
    chex.assert_trees_all_equal(
        outputs.data, (inputs + inputs.coords['lon']).data)

  def test_jit_function_with_xarray_dataset_arguments_and_return(self):
    foo = jnp.ones((3, 4), dtype=np.float32)
    bar = jnp.ones((2, 3, 4), dtype=np.float32)
    inputs = xarray_jax.Dataset(
        data_vars={'foo': (('lat', 'lon'), foo),
                   'bar': (('time', 'lat', 'lon'), bar)},
        coords={
            'time': np.arange(2),
            'lat': np.arange(3) * 10,
            'lon': np.arange(4) * 10})
    fn = jax.jit(lambda v: v + 1)
    _ = fn(inputs)
    outputs = fn(inputs)
    self.assertEqual({'foo', 'bar'}, outputs.data_vars.keys())
    self.assertEqual(inputs.foo.dims, outputs.foo.dims)
    self.assertEqual(inputs.bar.dims, outputs.bar.dims)
    chex.assert_trees_all_equal(outputs.coords, inputs.coords)

  def test_jit_function_with_xarray_datatree_arguments_and_return(self):
    parent_dataset = xarray_jax.Dataset(
        jax_coords={'time': xarray.Variable(('time',), np.arange(2))},
        coords={'lon': xarray.Variable(('lon',), np.arange(4) * 10)})

    bar = jnp.ones((2, 3, 4), dtype=np.float32)
    child_dataset = xarray_jax.Dataset(
        {'bar': (('time', 'lat', 'lon'), bar)},
        coords={'lat': np.arange(3)})

    inputs = xarray.DataTree(
        dataset=parent_dataset,
        children={'child': xarray.DataTree(dataset=child_dataset)})

    @jax.jit
    def fn(inputs):
      return xarray.DataTree(
          dataset=xarray_jax.assign_jax_coords(
              inputs.to_dataset(), time=inputs.time + 1),
          children={'child': xarray.DataTree(
              dataset=inputs.children['child'].to_dataset() + 1)})

    _ = fn(inputs)
    outputs = fn(inputs)
    self.assertEqual({'child'}, outputs.children.keys())
    self.assertEqual({'time', 'lon'}, outputs.coords.keys())
    self.assertIsInstance(
        outputs.coords['time'].data, jax.Array)
    self.assertEqual({'bar'}, outputs.child.data_vars.keys())

  def test_jit_function_with_dataset_and_jax_coords(self):
    foo = jnp.ones((3, 4), dtype=np.float32)
    bar = jnp.ones((2, 3, 4), dtype=np.float32)
    inputs = xarray_jax.Dataset(
        data_vars={'foo': (('lat', 'lon'), foo),
                   'bar': (('time', 'lat', 'lon'), bar)},
        coords={
            'time': np.arange(2),
            'lat': np.arange(3) * 10,
        },
        jax_coords={'lon': jnp.arange(4) * 10}
    )
    # Verify the jax_coord 'lon' retains jax data, and has not been created
    # as an index coordinate:
    self.assertIsInstance(inputs.coords['lon'].data, jax.Array)
    self.assertNotIn('lon', inputs.indexes)

    @jax.jit
    def fn(v):
      # The non-JAX coords are passed with numpy array data and an index:
      self.assertIsInstance(v.coords['lat'].data, np.ndarray)
      self.assertIn('lat', v.indexes)

      # The jax_coord is passed with JAX array data:
      self.assertIsInstance(v.coords['lon'].data, jax.Array)
      self.assertNotIn('lon', v.indexes)

      # Use the jax coord in the computation:
      v = v + v.coords['lon']

      # Return with an updated jax coord:
      return xarray_jax.assign_jax_coords(v, lon=v.coords['lon'] + 1)

    _ = fn(inputs)
    outputs = fn(inputs)

    # Verify the jax_coord 'lon' has jax data in the output too:
    self.assertIsInstance(
        outputs.coords['lon'].data, jax.Array)
    self.assertNotIn('lon', outputs.indexes)

    self.assertEqual(outputs.dims, inputs.dims)
    chex.assert_trees_all_equal(outputs.coords['lat'], inputs.coords['lat'])
    # Check our computations with the coordinate values worked:
    chex.assert_trees_all_equal(
        (outputs.coords['lon']).data,
        (inputs.coords['lon']+1).data,
    )
    outputs_dict = {key: outputs[key].data for key in outputs}
    inputs_and_inputs_coords_dict = {
        key: (inputs + inputs.coords['lon'])[key].data
        for key in inputs + inputs.coords['lon']
    }
    chex.assert_trees_all_equal(outputs_dict, inputs_and_inputs_coords_dict)

  def test_eval_shape_with_xarray(self):
    # This needs jax.ShapeDtypeStruct to be wrappable inside xarray types.
    function = jax.jit(lambda v: v + 1)
    inputs = xarray.Variable(
        ('lat', 'lon'), jnp.ones((3, 4), dtype=np.float32))
    output_shapes = jax.eval_shape(function, inputs)
    self.assertIsInstance(output_shapes, xarray.Variable)
    self.assertEqual(output_shapes.shape, (3, 4))
    self.assertEqual(output_shapes.dtype, np.float32)

  def test_assign_coords_arg_types(self):
    # Check we can assign coords with a variety of shorthands mirroring those
    # supported by xarray's own APIs:
    result = xarray_jax.assign_coords(
        xarray_jax.Dataset(),
        jax_coords={
            'a': np.arange(2),
            'b': jnp.arange(2),
            'c': [0, 1],
            'd': 123,
            'e': xarray.Variable(('e2',), np.arange(2)),
            'f': xarray_jax.DataArray(data=np.arange(2), dims=('f2',)),
            'g': (('g2',), np.arange(2)),
        })
    xarray.testing.assert_equal(
        result.a.variable, xarray.Variable(('a',), np.arange(2)))
    xarray.testing.assert_equal(
        jax.device_get(result.b.variable),
        xarray.Variable(('b',), np.arange(2)))
    xarray.testing.assert_equal(
        result.c.variable, xarray.Variable(('c',), np.arange(2)))
    xarray.testing.assert_equal(
        result.d.variable, xarray.Variable((), 123))
    xarray.testing.assert_equal(
        result.e.variable, xarray.Variable(('e2',), np.arange(2)))
    xarray.testing.assert_equal(
        result.f.variable, xarray.Variable(('f2',), np.arange(2)))
    xarray.testing.assert_equal(
        result.g.variable, xarray.Variable(('g2',), np.arange(2)))


if __name__ == '__main__':
  absltest.main()