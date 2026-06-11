API Reference
=============

.. currentmodule:: lax

Top-level
---------

The main entry point is :func:`compile`, which returns a :class:`Solver` bundle
containing all JIT-compiled observables.

.. autofunction:: compile

.. autofunction:: make_wavefunction_source

.. autoclass:: MeshSpec

.. autoclass:: ChannelSpec

.. autoclass:: Solver

lax.constants
-------------

Physical constants and mass-factor utilities.

.. automodule:: lax.constants
   :members:

lax.spectral
------------

Mesh-independent observables operating on a :class:`~lax.spectral.Spectrum`.

.. autoclass:: lax.spectral.Spectrum
   :members:

.. autoclass:: lax.spectral.CoupledChannelParameters
   :members:

.. autofunction:: lax.spectral.rmatrix_from_spectrum

.. autofunction:: lax.spectral.greens_from_spectrum

.. autofunction:: lax.spectral.wavefunction_internal_from_spectrum

.. autofunction:: lax.spectral.smatrix_from_R

.. autofunction:: lax.spectral.open_channel_smatrix_from_R

.. autofunction:: lax.spectral.phases_from_S

.. autofunction:: lax.spectral.coupled_channel_parameters_from_S

.. autofunction:: lax.spectral.pade_interpolate

lax.models
----------

Reusable interaction models and preset system parameters.

.. autoclass:: lax.models.RotorCoupledOpticalModel
   :members:

.. autoclass:: lax.models.RotorChannel
   :members:

.. autofunction:: lax.models.channels_from_rotor_model

.. autofunction:: lax.models.make_rotor_coupled_optical_potential

.. autofunction:: lax.models.open_channel_count

.. autofunction:: lax.models.first_column_amplitudes_and_phases

.. autofunction:: lax.models.woods_saxon_form_factor

.. autofunction:: lax.models.woods_saxon_derivative

.. autofunction:: lax.models.uniform_sphere_coulomb_potential

.. autofunction:: lax.models.rotor_coupling_coefficient

.. autofunction:: lax.models.reid_np_j1_channels

.. autofunction:: lax.models.interaction_from_reid_np_j1

.. autofunction:: lax.models.reid_soft_core_triplet_components

Internal modules
----------------

The modules below are part of the public API but are not typically imported
directly — they are accessed through :func:`compile` and the :class:`Solver` bundle.

.. rubric:: lax.boundary

.. autoclass:: lax.boundary.BoundaryValues

.. autoclass:: lax.boundary.Mesh

.. autoclass:: lax.boundary.OperatorMatrices

.. autofunction:: lax.boundary.compute_boundary_values

.. rubric:: lax.meshes

.. autofunction:: lax.meshes.build_mesh

.. rubric:: lax.propagate

.. autofunction:: lax.propagate.build_legendre_x_propagation
