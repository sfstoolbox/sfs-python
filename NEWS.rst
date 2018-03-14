Version History
===============

Version 0.4.0 (2018-03-14):
 * Driving functions in time domain for a plane wave, point source, and
   focused source
 * Image source model for a point source in a rectangular room
 * DelayedSignal class and as_delayed_signal()
 * Improvements to the documentation
 * Start using Jupyter notebooks for examples in documentation
 * Spherical Hankel function as util.spherical_hn2
 * Use spherical_jn, spherical_yn from scipy.special instead of sph_jnyn
 * Generalization of the modal order argument in mono.source.point_modal()
 * Rename util.normal_vector() to util.normalize_vector()
 * Add parameter max_order to NFCHOA driving functions
 * Add beta parameter to Kaiser tapering window
 * Fix clipping problem of sound field plots with matplotlib 2.1
 * Fix elevation in util.cart2sph
 * Fix tapering.tukey() for alpha=1

Version 0.3.1 (2016-04-08):
 * Fixed metadata of release

Version 0.3.0 (2016-04-08):
 * Dirichlet Green's function for the scattering of a line source at an edge
 * Driving functions for the synthesis of various virtual source types with
   edge-shaped arrays by the equivalent scattering appoach
 * Driving functions for the synthesis of focused sources by WFS
 * Several refactorings, bugfixes and other improvements

Version 0.2.0 (2015-12-11):
 * Ability to calculate and plot particle velocity and displacement fields
 * Several function name and parameter name changes
 * Several refactorings, bugfixes and other improvements

Version 0.1.1 (2015-10-08):
 * Fix missing `sfs.mono` subpackage in PyPI packages

Version 0.1.0 (2015-09-22):
   Initial release.
