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
