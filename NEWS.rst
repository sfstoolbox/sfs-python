Version History
===============

Version 0.5.0 (2019-03-18):
 * Switching to separate `sfs.plot2d` and `sfs.plot3d` for plotting functions
 * Move `sfs.util.displacement()` to `sfs.fd.displacement()`
 * Switch to keyword only arguments
 * New default driving function for `sfs.fd.wfs.point_25d()`
 * New driving function syntax, e.g. `sfs.fd.wfs.point_25d()`
 * Example for the sound field of a pulsating sphere
 * Add time domain NFC-HOA driving functions `sfs.td.nfchoa`
 * `sfs.fd.synthesize()`, `sfs.td.synthesize()` for soundfield superposition
 * Change `sfs.mono` to `sfs.fd` and `sfs.time` to `sfs.td`
 * Move source selection helpers to `sfs.util`
 * Use `sfs.default` object instead of `sfs.defs` submodule
 * Drop support for legacy Python 2.7

Version 0.4.0 (2018-03-14):
 * Driving functions in time domain for a plane wave, point source, and
   focused source
 * Image source model for a point source in a rectangular room
 * `sfs.util.DelayedSignal` class and `sfs.util.as_delayed_signal()`
 * Improvements to the documentation
 * Start using Jupyter notebooks for examples in documentation
 * Spherical Hankel function as `sfs.util.spherical_hn2()`
 * Use `scipy.special.spherical_jn`, `scipy.special.spherical_yn` instead of
   `scipy.special.sph_jnyn`
 * Generalization of the modal order argument in `sfs.mono.source.point_modal()`
 * Rename `sfs.util.normal_vector()` to `sfs.util.normalize_vector()`
 * Add parameter ``max_order`` to NFCHOA driving functions
 * Add ``beta`` parameter to Kaiser tapering window
 * Fix clipping problem of sound field plots with matplotlib 2.1
 * Fix elevation in `sfs.util.cart2sph()`
 * Fix `sfs.tapering.tukey()` for ``alpha=1``

Version 0.3.1 (2016-04-08):
 * Fixed metadata of release

Version 0.3.0 (2016-04-08):
 * Dirichlet Green's function for the scattering of a line source at an edge
 * Driving functions for the synthesis of various virtual source types with
   edge-shaped arrays by the equivalent scattering appoach
 * Driving functions for the synthesis of focused sources by WFS

Version 0.2.0 (2015-12-11):
 * Ability to calculate and plot particle velocity and displacement fields
 * Several function name and parameter name changes

Version 0.1.1 (2015-10-08):
 * Fix missing `sfs.mono` subpackage in PyPI packages

Version 0.1.0 (2015-09-22):
   Initial release.
