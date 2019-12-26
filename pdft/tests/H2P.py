import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

bondlength = 5

Full_Molec =  psi4.geometry("""
nocom
noreorient
1 2
H  -%f 0.0 0
@H  0 0.5 0
@H  0 -0.5 0
H  %f 0.0 0
units bohr
symmetry c1
"""% (bondlength / 2, bondlength / 2))

Monomer_1 =  psi4.geometry("""
nocom
noreorient
@H  -%f 0.0 0
@H  0 0.5 0
@H  0 -0.5 0
H  %f 0.0 0
units bohr
symmetry c1
"""% (bondlength / 2, bondlength / 2))

Monomer_2 =  psi4.geometry("""
nocom
noreorient
H  -%f 0.0 0
@H  0 0.5 0
@H  0 -0.5 0
@H  %f 0.0 0
units bohr
symmetry c1
"""% (bondlength / 2, bondlength / 2))

Full_Molec.set_name("H2P")

#Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 110,
    # 'DFT_RADIAL_POINTS':    5,
    'REFERENCE' : 'UKS'
})

#Make fragment calculations:
f1  = pdft.U_Molecule(Monomer_2,  "cc-pvdz", "SVWN", omega=0.5)
f2  = pdft.U_Molecule(Monomer_1,  "cc-pvdz", "SVWN", omega=0.5)
mol = pdft.U_Molecule(Full_Molec, "cc-pvdz", "SVWN")

#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
rho_conv, ep_conv = pdfter.find_vp_response(maxiter=60, beta=1, atol=1e-5)
#%%
# pdfter.get_energies()
#%%
# #%% Plotting Cubic
# #Set the box lenght and grid fineness.
# L = [8.0,  8.0, 8.0]
# D = [0.2, 0.2, 0.2]
# # Plot file
# O, N =  libcubeprop.build_grid(mol.wfn, L, D)
# block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
# f, (ax1, ax2) = plt.subplots(1, 2)
# vp_cube = libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, vp)

#%%

vp_grid = mol.to_grid(pdfter.vp[0].np)
pdft.plot1d_x(vp_grid, mol.Vpot, title="vp", dimmer_length=bondlength)

#
# fig1 = plt.figure(num=1, figsize=(16, 12))
# plt.plot(rho_conv, figure=fig1)
# plt.xlabel(r"iteration")
# plt.ylabel(r"$\int |\rho_{whole} - \sum_{fragment} \rho|$")
# plt.title(r"$H2^+$ w/ response method ")
# fig1.savefig("rho")
# plt.show()
# fig2 = plt.figure(num=2, figsize=(16, 12))
# plt.plot(ep_conv, figure=fig2)
# plt.xlabel(r"iteration")
# plt.ylabel(r"Ep")
# plt.title(r"$H2^+$ w/ response method ")
# fig2.savefig("Ep")
# plt.show()

# %%
