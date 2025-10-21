# xarray_jax
[![PyPI](https://img.shields.io/pypi/v/xarray-jax.svg)](https://pypi.org/project/xarray-jax/)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

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

## What is xarray_jax?

`JAX` is a high-performance library designed for numerical computation and machine learning, operating primarily on a specific data structure known as a **PyTree**. Standard `xarray` objects (`DataArray` and `Dataset`), while powerful for labeled scientific data, are not native PyTrees, preventing their direct use within `JAX`'s core transformations (`jit`, `grad`, `vmap`, etc.).

**This library solves that problem.** It registers `xarray` data structures as custom JAX PyTrees. This allows `JAX` to seamlessly **flatten** `xarray` objects into their raw arrays for accelerated computation and then **unflatten** the results back into fully labeled `xarray` objects, preserving critical metadata like dimension names and coordinates.

Furthermore, `xarray-jax` ensures robustness, especially when used with advanced frameworks like Flax or Haiku. It internally uses a `NonArrayLeafWrapper` to correctly handle non-array data (like numbers or `None`) that might appear within the PyTree structure during complex `JAX` transformations, preventing unexpected errors.



## Installation

```bash
pip install xarray-jax
```

## Quick Start

Here's a minimal example showing how to apply JAX's just-in-time (JIT) compilation to a function that operates directly on an `xarray.DataArray` containing `jax.numpy` data.

*Example :*
```python
import jax
import jax.numpy as jnp
import xarray as xr
import numpy as np #import numpy for datetime64
import xarray_jax

# 1. Create a standard xarray.DataArray with JAX data
#   Use np.datetime64 for time coordinates as recommended practice
temperature = xr.DataArray(
    data=jnp.array([[20.5, 21.2, 22.1],
                   [18.3, 19.7, 20.8]]),
    dims=('time', 'location'),
    coords={
        'time': np.array(['2023-01-01', '2023-01-02'], dtype='datetime64[D]'),
        'location': ['NYC', 'LA', 'Chicago']
    }
)

# 2. Define a pure Python function that works with the DataArray
#    This function can use standard xarray methods like .mean()
def process_temperature(temp_data):
    fahrenheit = temp_data * 9/5 + 32
    return fahrenheit.mean(dim='location')

# 3. JIT-compile the function using jax.jit
#    This would fail without xarray-jax registering xarray objects as PyTrees
jitted_process = jax.jit(process_temperature)

# 4. Execute the compiled function with the xarray object
result = jitted_process(temperature)

print("JIT compilation successful!")
print(result)
# Expected Output:
# <xarray.DataArray (time: 2)>
# array([70.28    , 67.27999 ], dtype=float32)
# Coordinates:
#   * time     (time) datetime64[ns] 2023-01-01 2023-01-02
```
## Using xarray in JAX Transformations

The primary use case for `xarray_jax` is enabling JAX transformations on functions that work with xarray data structures. Since JAX arrays are now directly supported by xarray, you can create xarray objects with JAX arrays and use them directly.

### Computing Gradients with `jax.grad`

You can compute gradients through functions operating on `xarray.DataArray` objects, provided the function returns a raw JAX scalar. Use `xarray_jax.jax_data` to extract the underlying JAX array when needed.

*Example :*

```python
# Assume 'temperature' DataArray exists from the Quick Start example

# Define a function that returns a scalar for gradient computation
def temperature_loss(temp_data):
    target = 20.0 # Simple loss: penalize deviation from 20°C
    # Note: The differentiable loss function must return a raw JAX scalar
    loss_xarray = ((temp_data - target) ** 2).sum()
    return xarray_jax.jax_data(loss_xarray) # Extract JAX scalar

# Compute gradients with respect to the temperature data
grad_fn = jax.grad(temperature_loss)
gradients = grad_fn(temperature) # Pass the original xarray object

print(gradients)
# Expected Output:
# <xarray.DataArray (time: 2, location: 3)>
# array([[ 1.       ,  2.4000015,  4.200001 ],
#        [-3.4000015, -0.5999985,  1.5999985]], dtype=float32)
# Coordinates:
#   * time      (time) datetime64[ns] 2023-01-01 2023-01-02
#   * location  (location) <U7 'NYC' 'LA' 'Chicago'
```

### Working with Datasets
---

JAX transformations also work seamlessly with `xarray.Dataset` objects.

*Example :*

```python
# Create a Dataset with multiple variables
weather_data = xr.Dataset(
    {
        'temperature': (['time', 'location'],
                       jnp.array([[20.5, 21.2], [18.3, 19.7]])),
        'humidity': (['time', 'location'],
                    jnp.array([[0.65, 0.72], [0.58, 0.69]]))
    },
    coords={
        'time': ['morning', 'evening'],
        'location': ['NYC', 'LA']
    }
)

@jax.jit
def weather_model(data):
    # Compute heat index approximation
    temp_f = data.temperature * 9/5 + 32
    heat_index = temp_f + 0.5 * (data.humidity * 100 - 10)
    # Use standard xarray method to assign a new variable
    return data.assign(heat_index=heat_index)

result_dataset = weather_model(weather_data)
print(result_dataset.heat_index)
# Expected Output:
# <xarray.DataArray 'heat_index' (time: 2, location: 2)>
# array([[96.15     , 99.7      ],
#        [92.44     , 96.9600067]], dtype=float32)
# Coordinates:
#   * time      (time) <U7 'morning' 'evening'
#   * location  (location) <U3 'NYC' 'LA'
```


## Advanced JAX Operations with Named Dimensions

`xarray_jax` provides specialized wrappers for certain JAX operations to work directly with xarray's named dimensions, making parallelization and iteration more intuitive in specific contexts.

### Vectorized Mapping with `xarray_jax.vmap`

**Note:** While `xarray_jax.vmap` exists, it is generally **not needed** for standard xarray usage due to xarray's built-in broadcasting. Furthermore, `vmap` is **not the canonical method for SPMD or sharding** in modern JAX (see the Sharding section below). This function is mainly for niche use cases, such as compatibility with legacy code requiring named axes for collectives or when vectorizing specific non-broadcastable JAX primitives.

`vmap` vectorizes a function along a named xarray dimension, acting as a thin, dimension-name-aware wrapper around `jax.vmap`.

*Constraints:*
* The mapped dimension `dim` must be the leading axis for all relevant xarray variables and JAX arrays passed to the function.
* Uses `in_axes=0, out_axes=0` strictly (no broadcasting of non-mapped leaves).
* Static coordinates along `dim` are not available inside the function; prefer `jax_coords` if per-example coordinates are needed.
* Optional `axis_name`, `axis_size`, and `spmd_axis_name` are forwarded to `jax.vmap` for advanced use cases like collectives or legacy sharding interop ([see jax.vmap docs](https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html)).

*Example :*
```python
import xarray
import jax
import jax.numpy as jnp
import xarray_jax

batch = 5
ds = xarray.Dataset(
    {
        'foo': (('batch', 'lat', 'lon'), jnp.zeros((batch, 3, 4), dtype=jnp.float32)),
        'bar': (('batch', 'time', 'lat', 'lon'), jnp.zeros((batch, 2, 3, 4), dtype=jnp.float32)),
    },
    coords={'lat': jnp.arange(3), 'lon': jnp.arange(4)}
)

def step(d):  # 'batch' dimension is removed inside the vmapped function
  assert 'batch' not in d.dims
  return d + 1

# Vectorize the 'step' function along the 'batch' dimension
vmapped_step = xarray_jax.vmap(step, dim='batch')
result = vmapped_step(ds)
print(result.dims) # {'batch': 5, 'lat': 3, 'lon': 4, 'time': 2}
```
## Parallel Mapping with `xarray_jax.pmap`

**Note:** Similar to `vmap`, `jax.pmap` is **no longer the main or most flexible way to achieve sharding** in JAX. Modern approaches involve directly specifying sharding on arrays (see the Sharding section below). `xarray_jax.pmap` is provided primarily for compatibility or specific device-level parallel compulations patterns outside of typical sharding.

The `pmap` function allows you to parallelize computation across multiple devices (e.g., GPUs or TPUs) using a named dimension:

*Example :*
```python
# Create data with a 'device' dimension matching the number of local devices
devices_count = jax.local_device_count()
parallel_data = xr.DataArray(
    data=jnp.ones((devices_count, 10, 5)),
    dims=('device', 'time', 'location'),
    coords={
        'time': range(10),
        'location': range(5)
    }
)

def process_chunk(data_chunk):
    # This function operates on a slice of data on a single device.
    # The 'device' dimension is not present here.
    assert 'device' not in data_chunk.dims
    return data_chunk * 2

# Parallelize the function across devices using the named dimension
parallel_fn = xarray_jax.pmap(process_chunk, dim='device')
result = parallel_fn(parallel_data)

print(result.dims)  # ('device', 'time', 'location')
```
## Sharding (Modern Approach)

The canonical way to perform parallel computation and distribute data across multiple devices (SPMD/sharding) in modern JAX involves using **device meshes** and attaching **sharding information directly to arrays**. This approach is generally preferred over legacy methods like `pmap` or `vmap` for sharding. `xarray_jax` facilitates this by integrating well with `jax.sharding` primitives, particularly through the `tree_map_with_dims` utility.

Here's an example demonstrating how to shard an `xarray.Dataset` along the 'batch' dimension using `jax.sharding.NamedSharding`

*Example :*
```python
import jax
import jax.numpy as jnp
import xarray as xr
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import xarray_jax

# 1. Create a device mesh
#    (Example: Use 2 devices if available, map 'batch' dimension to mesh axis)
try:
    # Attempt to create a mesh, ideally with multiple devices
    num_devices = jax.local_device_count()
    # Ensure batch size is suitable for sharding across devices
    batch_size_per_device = 4
    batch = num_devices * batch_size_per_device

    devices = mesh_utils.create_device_mesh((num_devices,))
    # Name the mesh axis 'batch' to align with the dimension name
    mesh = Mesh(devices, axis_names=('batch',))
    print(f"Using mesh: {mesh.devices.shape} devices, axis names: {mesh.axis_names}")
    multi_device = True

except RuntimeError as e:
    # Fallback for environments with only 1 device (like CPU/single GPU)
    print(f"Could not create multi-device mesh ({e}). Running on single device.")
    mesh = Mesh(jax.devices(), axis_names=('batch',)) # Dummy mesh for single-device fallback (no real sharding)
    multi_device = False
    batch = 8 # Example batch size for single device

# 2. Define a sharding rule function
#    This function applies sharding based on dimension names
def apply_sharding(arr, dims):
    if dims and dims[0] == 'batch' and multi_device:
        # Shard the first dimension ('batch') across the 'batch' mesh axis
        sharding = jax.sharding.NamedSharding(mesh, P('batch',))
    else:
        # Replicate arrays that don't start with 'batch' or if on single device
        sharding = jax.sharding.NamedSharding(mesh, P()) # Replicated sharding
    
    # Place the array onto devices with the specified sharding
    return jax.device_put(arr, sharding)

# 3. Prepare your xarray data
ds = xr.Dataset(
    {'foo': (('batch', 'lat', 'lon'), jnp.zeros((batch, 3, 4)))},
    coords={'lat': jnp.arange(3), 'lon': jnp.arange(4)}
)

# 4. Apply sharding using tree_map_with_dims
#    This iterates through the dataset and applies the sharding rule
sharded_ds = xarray_jax.tree_map_with_dims(apply_sharding, ds)

# Verify the sharding (optional, shows how data is distributed)
print(f"Sharding applied to 'foo': {sharded_ds.foo.data.sharding}")

# 5. Pass sharded data directly to a jitted function
#    No vmap or pmap needed! JAX handles the SPMD computation.
@jax.jit
def process_sharded_data(data):
    # Your computation here operates directly on the sharded data
    return data * 2 + 1

# Execute the JITted function
result = process_sharded_data(sharded_ds)

print("Sharded computation successful.")
# The output sharding generally matches the input sharding
print(f"Output sharding of 'foo': {result.foo.data.sharding}")
```
## Utility Functions

While `xarray-jax` allows direct use of `xarray` objects in JAX transformations, you might occasionally need to interact with the underlying array data.

### Data Access Recommendation

For accessing the raw array data within an `xarray.DataArray` or `xarray.Variable`, the **recommended approach** is to use the standard xarray API:

```python
# Assume 'temperature' is an xarray.DataArray with JAX data
jax_array = temperature.data
```
### Extraction Helpers(`unwrap_data`, `jax_data`, etc.)

Funtions like `unwrap_data` and `jax_data` exist primarily for **backward compatibility** and **internal library use** (e.g., handling non-array leaves within the PyTree logic). Most users should **prefer the standard** `.data` **attribute** for clarity and consistency with the broader xarray ecosystem.

These helper functions might be useful only in specific advanced scenarios:

* `unwrap_data`: Flexibly extracts the underlying array, whether it's a JAX `Array` or NumPy `ndarray`. Potentially useful for type-agnostic code or when moving data outside the JAX ecosystem (e.g., plotting), although accessing `.data` often suffices. 

* `jax_data`: Strictly extracts data only if it's a JAX `Array`, raising an error otherwise. Its primary use case is inside JAX-transformed functions (like loss functions for `jax.grad`) to ensure you have a JAX array or tracer, which is sometimes required by JAX primitives.

*Example :*

```python
# (Usually, just use .data like below, but showing helpers for completeness)

import xarray as xr
import jax.numpy as jnp
import numpy as np
import xarray_jax

# Create an xarray DataArray using NumPy
numpy_backed_data = xr.DataArray(np.array([1, 2, 3]))

# Create an xarray DataArray using JAX numpy
jax_backed_data = xr.DataArray(jnp.array([1, 2, 3]))

# --- Method 1: The Recommended Way (Using .data) ---
# This directly accesses the underlying array, keeping things simple.
raw_jax_array = jax_backed_data.data
print(f"Using .data (Recommended): {type(raw_jax_array)}")

# --- Method 2: Using Helper Functions (Less Common) ---
# 'unwrap_data' works for both NumPy and JAX arrays inside xarray
raw_numpy_array_helper = xarray_jax.unwrap_data(numpy_backed_data)
raw_jax_array_helper = xarray_jax.unwrap_data(jax_backed_data)
print(f"Using unwrap_data (NumPy-backed): {type(raw_numpy_array_helper)}")
print(f"Using unwrap_data (JAX-backed): {type(raw_jax_array_helper)}")

# 'jax_data' ONLY works if the data inside is already a JAX array
# Useful inside JIT/Grad functions to be sure you have a JAX array
raw_jax_array_strict = xarray_jax.jax_data(jax_backed_data)
print(f"Using jax_data (JAX-backed): {type(raw_jax_array_strict)}")

# This next line would cause an error because the data inside isn't JAX:
# error_example = xarray_jax.jax_data(numpy_backed_data)
```
Other helpers like `unwrap_vars`, `jax_vars`, and `unwrap_coords` exist for extracting data and coordinates from `xarray.Dataset` objects, following similar principles. Prefer direct attribute access (`.data_vars`, `.coords`) where possible.

Standard `xarray` coordinates (`coords`) are treated by JAX as **static**, compile-time constants. For scenarios requiring coordinates that are dynamic, traceable JAX arrays (e.g., coordinates that change during an optimization process), `xarray-jax` provides the **`jax_coords`** parameter in its custom `DataArray` and `Dataset` constructors.

This feature should currently be considered **experimental**. While relied upon by some DeepMind research models (like GenCast), it may be replaced in the future with a different approach (e.g., automatically treating all non-index coordinates as dynamic).

Therefore, it is **recommended** that most users prefer the standard `xarray.DataArray` and `xarray.Dataset` constructors and use standard `coords`, unless there is a specific need for dynamic, traceable coordinates.


*Example :*
```python
import xarray
import jax
import jax.numpy as jnp
import xarray_jax

batch = 5
# Dynamic time values computed with JAX
time_values = jnp.linspace(0.0, 1.0, batch)

# Use xarray_jax.Dataset to define 'time' as a jax_coord
ds_dynamic = xarray_jax.Dataset(
    {'data': (('batch', 'x'), jnp.ones((batch, 3)))},
    coords={'x': [0, 1, 2]},
    # 'time' is now a traceable JAX array associated with the 'batch' dimension
    jax_coords={'time': xarray.Variable(('batch',), time_values)}
)

@jax.jit
def process_dynamic(d):
    # This JITted function can perform calculations
    # using the dynamic 'time' coordinate because it's a JAX array.
    return d.data * d.coords['time']

result_dynamic = process_dynamic(ds_dynamic)
print(result_dynamic.dims) # ('batch', 'x')
# Verify that the 'time' coordinate data is still a JAX array
print(f"Type of time coord data: {type(result_dynamic.coords['time'].data)}")
```
## Framework Support

A core design goal of `xarray-jax` is to ensure robust compatibility with the broader JAX ecosystem, including higher-level neural network libraries like [Flax](https://flax.readthedocs.io/en/latest/) and [Haiku](https://dm-haiku.readthedocs.io/en/latest/). These frameworks often perform complex manipulations on PyTrees during operations like model initialization or compilation (`jax.jit`).

A key challenge arises because the JAX PyTree contract allows *any* data type to be a leaf node, not just arrays. Transformations can introduce non-array leaves such as `None`, integers, booleans, or internal JAX metadata objects (like `jax.ShapeDtypeStruct` used during tracing). However, standard `xarray` constructors expect array-like data.

To bridge this gap and prevent errors, `xarray-jax` internally uses a `NonArrayLeafWrapper`. This utility class wraps any non-array leaf value encountered during the PyTree flattening/unflattening process. It implements the minimal properties (`.shape`, `.dtype`) required to "look like" a zero-dimensional array (duck typing), satisfying `xarray`'s expectations without actually converting the leaf into an array.

This wrapper ensures that JAX transformations applied to functions using `xarray` objects remain robust even when frameworks introduce non-array leaves into the PyTree structure, allowing seamless integration with advanced JAX patterns.

## Scan

`jax.lax.scan` is a powerful primitive for efficiently executing loops with carry-over state, commonly used in recurrent neural networks or time-series simulations. `xarray_jax` provides a wrapper, **`xarray_jax.scan`**, that allows you to use this functionality directly with `xarray` objects, scanning along a named dimension.

The `xarray_jax.scan` wrapper behaves like `jax.lax.scan` but automatically handles named dimensions and reconstructs the `xarray` structure on output.

This wrapper handles the details of transposing the specified dimension (`dim`) to the leading axis before passing it to `jax.lax.scan` and correctly reconstructing the `xarray` object with the scanned dimension afterwards.

*Constraints:*
* The scanned dimension `dim` must exist in all `xarray` objects within the input `xs`.
* Static coordinates along `dim` in the input `xs` will not be available inside the scan function `f` and are not automatically restored on the output `ys`. Prefer `jax_coords` if per-step coordinates are needed dynamically within `f`.

*Example :*
```python
import jax
import jax.numpy as jnp
import xarray as xr
import xarray_jax

# Initial state (carry)
initial_state = jnp.float32(0.0)

# Input data with a 'time' dimension to scan over
input_data = xr.DataArray(
    data=jnp.arange(1, 6, dtype=jnp.float32),
    dims=('time',),
    coords={'time': jnp.arange(5)}
)

# Define the function to apply at each step
# Takes (carry, x_slice) and returns (new_carry, y_slice)
def scan_body(carry, x_slice):
    # x_slice is the data for the current time step (no 'time' dim)
    new_carry = carry + x_slice.data # Access raw JAX array
    y_slice = new_carry * 2 # Example output for this step
    return new_carry, y_slice

# Perform the scan along the 'time' dimension
final_state, output_y = xarray_jax.scan(scan_body, initial_state, dim='time', xs=input_data)

print("Final State (Carry):", final_state)
print("\nOutput (ys):")
print(output_y)
# Expected Output:
# Final State (Carry): 15.0
#
# Output (ys):
# <xarray.DataArray (time: 5)>
# array([ 2.,  6., 12., 20., 30.], dtype=float32)
# Coordinates:
#   * time     (time) int32 0 1 2 3 4
```