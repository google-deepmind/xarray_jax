# xarray_jax

This library supports using [xarray](https://docs.xarray.dev/en/stable/)
datatypes together with the
[JAX](https://docs.jax.dev/en/latest/quickstart.html) library.

## Current status

This project has been spun off from the
[xarray_jax utility in the graphcast project](https://github.com/google-deepmind/graphcast/blob/main/graphcast/xarray_jax.py).

We plan to make some improvements to it before migrating existing uses
(including graphcast) to it. These improvements are not yet complete and so
we don't advise migrating to this code yet.

We will aim to make improvements largely backwards-compatible with
`graphcast.xarray_jax`, but they may not be entirely so.


## **What is `xarray_jax`?**

`xarray_jax` is a Python library for accelerator-oriented Xarray computation and program transformation, designed for high-performance numerical computing and large-scale machine learning. It allows you to use `xarray.DataArray` and `xarray.Dataset` objects directly within JAX transformations like `jax.jit`, `jax.grad`, and more. It also provides named-dimension wrappers for `jax.vmap`, `jax.pmap`, and `jax.lax.scan`.

By registering Xarray data structures as PyTrees, JAX knows how to flatten Xarray objects into their constituent arrays for computation, and then unflatten the results back into properly structured Xarray objects with the correct dimension names and coordinates. You can pass Xarray objects to any JAX transformation function and get Xarray objects back out.

## Using Xarray in JAX Transformations

The primary use case for `xarray_jax` is enabling JAX transformations on functions that work with Xarray data structures. Since JAX arrays are now directly supported by Xarray, you can create Xarray objects with JAX arrays and directly use them in JAX transformations.

### Basic Usage with `jax.jit`

```python
import jax
import jax.numpy as jnp
import xarray
import xarray_jax

temperature = xarray.DataArray(
    data=jnp.array([[20.5, 21.2, 22.1],
                    [18.3, 19.7, 20.8]]),
    dims=('time', 'location'),
    coords={
        'time': ['2023-01-01', '2023-01-02'],
        'location': ['NYC', 'LA', 'Chicago']
    }
)

# Operates directly on the DataArray
def process_temperature(temp_data):
    fahrenheit = temp_data * 9/5 + 32
    return fahrenheit.mean(dim='location')

jitted_process = jax.jit(process_temperature)
result = jitted_process(temperature) # Operates directly on the DataArray
print(result)
# <xarray.DataArray (time: 2)>
# array([70.28, 67.28], dtype=float32)
# Coordinates:
#   * time     (time) <U10 '2023-01-01' '2023-01-02'
```

### Computing Gradients with `jax.grad`

```python
# Define a function that returns a scalar for gradient computation
def temperature_loss(temp_data):
    target = 20.0 # Simple loss: penalize deviation from 20°C
    # Note: The differentiable loss function must return a raw JAX scalar
    loss_xarray = ((temp_data - target) ** 2).sum()
    return xarray_jax.jax_data(loss_xarray)

# Compute gradients with respect to the temperature data
grad_fn = jax.grad(temperature_loss)
gradients = grad_fn(temperature)

print(gradients)
# <xarray.DataArray (time: 2, location: 3)>
# array([[ 1.       ,  2.4000015,  4.200001 ],
#        [-3.4000015, -0.5999985,  1.5999985]], dtype=float32)
# Coordinates:
#   * time      (time) <U10 '2023-01-01' '2023-01-02'
#   * location  (location) <U7 'NYC' 'LA' 'Chicago'
```

### Working with Datasets

```python
# Create a Dataset with multiple variables
weather_data = xarray.Dataset({
    'temperature': (['time', 'location'],
                   jnp.array([[20.5, 21.2], [18.3, 19.7]])),
    'humidity': (['time', 'location'],
                jnp.array([[0.65, 0.72], [0.58, 0.69]]))
}, coords={
    'time': ['morning', 'evening'],
    'location': ['NYC', 'LA']
})

@jax.jit
def weather_model(data):
    # Compute heat index approximation
    temp_f = data.temperature * 9/5 + 32
    heat_index = temp_f + 0.5 * (data.humidity * 100 - 10)
    return data.assign(heat_index=heat_index)

result = weather_model(weather_data)
print(result.heat_index)
```

## Advanced JAX Operations with Named Dimensions

`xarray_jax` provides specialized wrappers for JAX operations using Xarray's named dimensions, making parallelization and iteration more intuitive.

### Vectorized Mapping with `xarray_jax.vmap`

While Xarray's dimension-aware broadcasting automatically handles many vectorization use cases (e.g., `ds + 1`), `xarray_jax.vmap` provides an explicit way to vectorize functions that are not trivially broadcastable and serves as the primary way to use advanced JAX features like parallel collectives (`psum`) and SPMD sharding with named dimensions.

`vmap` vectorizes a function along a named Xarray dimension, acting as a thin, dimension-name-aware wrapper around `jax.vmap`.

Constraints:

- The mapped dimension `dim` must be the leading axis for all Xarray variables (including `jax_coords`) and any plain JAX arrays passed to the function.
- Uses `in_axes=0, out_axes=0`. Broadcasting non-batched leaves is not supported.
- Static coordinates along the mapped dimension (`dim`) are not available inside the function and are not automatically restored in the output. Prefer `jax_coords` for per-example coordinates that need to be available during the vectorized computation.
- Optional `axis_name` allows collectives over the vmapped axis.
- Optional `axis_size` and `spmd_axis_name` are forwarded to JAX’s API for explicit axis sizing and SPMD sharding interop. See [jax.vmap docs](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html).

Example:

```python
import xarray
import jax
import jax.numpy as jnp
import xarray_jax

batch = 5
ds = xarray_jax.Dataset({
    'foo': (('batch', 'lat', 'lon'), jnp.zeros((batch, 3, 4), dtype=jnp.float32)),
    'bar': (('batch', 'time', 'lat', 'lon'), jnp.zeros((batch, 2, 3, 4), dtype=jnp.float32)),
}, coords={'lat': jnp.arange(3), 'lon': jnp.arange(4)})

def step(d):
  # 'batch' is removed inside the vmapped function
  assert 'batch' not in d.dims
  return d + 1

vmapped_step = xarray_jax.vmap(step, dim='batch')
result = vmapped_step(ds)
```

With `jax_coords`:

```python
time = jnp.zeros((batch, 2), dtype=jnp.float32)
ds = xarray_jax.Dataset(
  {'bar': (('batch', 'time', 'lat', 'lon'), jnp.zeros((batch, 2, 3, 4)))},
  coords={'lat': jnp.arange(3), 'lon': jnp.arange(4)},
  jax_coords={'time': xarray.Variable(('batch', 'time'), time)},
)

def step(d):
  # jax_coord 'time' is a JAX array, not an index coordinate
  assert isinstance(d.coords['time'].data, jax.Array)
  assert 'batch' not in d.coords['time'].dims
  return d + 1

vmapped_step = xarray_jax.vmap(step, dim='batch')
result = vmapped_step(ds)
```

Sharding interop:

- To integrate with sharding (e.g., NNX or `jax.sharding`), you can do one of the following:
    - Provide `spmd_axis_name` in `xarray_jax.vmap` for SPMD-style named axis propagation
    - Use `xarray_jax.tree_map_with_dims` to attach `with_sharding_constraint` or `NamedSharding` to leaves based on their dims.
- Example snippet applying sharding by dims:

```python
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P

devices = mesh_utils.create_device_mesh((1, 1))
with Mesh(devices, ('batch', 'model')):
  def apply_sharding(arr, dims):
    if dims is None:
      return arr
    if dims and dims[0] == 'batch':
      return jax.lax.with_sharding_constraint(
          arr, P('batch', *([None] * (arr.ndim - 1)))
      )
    return arr

  sharded = xarray_jax.tree_map_with_dims(apply_sharding, ds)
  vmapped = xarray_jax.vmap(step,
                            dim='batch',
                            axis_name='batch',
                            spmd_axis_name='batch'
                            )
  out = vmapped(sharded)
```

### Parallel Mapping with `xarray_jax.pmap`

The `pmap` function allows you to parallelize computation across multiple devices while working with named dimensions:

```python
import xarray_jax

# Create data with a 'device' dimension for parallel processing
devices = jax.local_device_count()
parallel_data = xarray.DataArray(
    data=jnp.ones((devices, 10, 5)),
    dims=('device', 'time', 'location'),
    coords={
        'time': range(10),
        'location': range(5)
    }
)

def process_chunk(data_chunk):
    # This function operates on a slice of data on a single device.
    # The 'device' dimension is not present here.
    return data_chunk * 2

# Parallelize the function across devices using the named dimension
parallel_fn = xarray_jax.pmap(process_chunk, dim='device')
result = parallel_fn(parallel_data)

print(result.dims)  # ('device', 'time', 'location')
```

## Utility Functions

### Data Extraction: `jax_data` vs. `unwrap_data`

When you need to extract the underlying array data from an Xarray object, `xarray_jax` provides two main functions: `unwrap_data` and `jax_data`.

### `unwrap_data`: For Flexibility

The `unwrap_data` function is the more general-purpose of the two. It extracts the underlying array data regardless of whether it's a JAX `Array` or a NumPy `ndarray`.

**When to use `unwrap_data`:**

- **Array-type Agnostic Code:** When writing functions that should work directly with Xarray objects containing either JAX or NumPy arrays.
- **Data Inspection:** When you want to inspect or use the data outside of JAX's ecosystem, for example, for plotting with Matplotlib or saving to a file.

```python
# unwrap_data works with both JAX and NumPy-backed DataArrays
numpy_da = xarray.DataArray(np.array([1, 2, 3]))
jax_da = xarray.DataArray(jnp.array([1, 2, 3]))

numpy_array = xarray_jax.unwrap_data(numpy_da)  # Returns ndarray
jax_array_unwrapped = xarray_jax.unwrap_data(jax_da)   # Returns jax.Array
```

### `jax_data`: For JAX-Specific Operations

The `jax_data` function is stricter. It only succeeds if the underlying data is a JAX `Array`. If the data is a NumPy array or any other type, it will raise a `TypeError`.

**When to use `jax_data`:**

- **Ensuring JAX Compatibility:** Use it inside JAX-transformed functions (`jit`, `grad`, `vmap`, etc.) to guarantee you are working with a JAX array or tracer, which is often a requirement. For example, a loss function passed to `jax.grad` must return a scalar JAX array.
- **Assertions:** To assert that your data has remained within the JAX ecosystem throughout a series of computations.

```python
def temperature_loss(temp_data):
    # A loss function for jax.grad must return a raw JAX scalar.
    # Using jax_data ensures this.
    loss_xarray = ((temp_data - 20.0) ** 2).sum()
    return xarray_jax.jax_data(loss_xarray)

grad_fn = jax.grad(temperature_loss)
gradients = grad_fn(temperature) # temperature is an xarray.DataArray with JAX data

# This would fail:
numpy_da = xarray.DataArray(np.array([1, 2, 3]))
# xarray_jax.jax_data(numpy_da)  # Raises TypeError
```

In summary:

| Function | Input Array Type | Output | Use Case |
| --- | --- | --- | --- |
| `unwrap_data` | JAX or NumPy | JAX or NumPy array | Flexible, type-agnostic code |
| `jax_data` | JAX only | JAX array | Stricter, for JAX-specific contexts |

### Other Extraction Utilities

`xarray_jax` also provides helpers for working with `Dataset` variables and coordinates:

```python
# Extract all data variables from a Dataset as a dictionary of JAX arrays
# (will error if any variable is not a JAX array)
var_dict = xarray_jax.jax_vars(weather_data)
# Returns: {'temperature': jax_array1, 'humidity': jax_array2}

# Extract all data variables flexibly (JAX or NumPy)
unwrapped_vars = xarray_jax.unwrap_vars(weather_data)

# Extract coordinates (JAX or NumPy)
coord_dict = xarray_jax.unwrap_coords(weather_data)
```

### Working with Variables

You can also work with `xarray.Variable` objects directly, as they are also registered as PyTrees.

```python
# Work with Variables directly (Variables are also PyTree-registered)
temp_var = xarray.Variable(('time', 'location'),
                      jnp.array([[20.5, 21.2], [18.3, 19.7]]))

@jax.jit
def process_variable(var):
    return var + 1

result_var = process_variable(temp_var)
```

## Dynamic Coordinates

The custom `xarray_jax.DataArray` and `xarray_jax.Dataset` constructors support a `jax_coords` parameter for defining dynamic coordinates. Unlike standard `coords`, which are treated as static, compile-time constants by JAX, `jax_coords` are treated as dynamic, traceable JAX arrays. This feature is provided for backward compatibility with Google DeepMind's Gencast research models and can be ignored by most users, who should prefer the standard `xarray.DataArray` and `xarray.Dataset` constructors.