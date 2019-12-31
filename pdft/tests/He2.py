import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

bondlength = 5

Full_Molec =  psi4.geometry( """
nocom
noreorient
He %f 0.0 0.00
@H  0 2.5 0
@H  0 -2.5 0
He -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Monomer_1 =  psi4.geometry("""
nocom
noreorient
He %f 0.0 0.00
@H  0 2.5 0
@H  0 -2.5 0
@He -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Monomer_2 =  psi4.geometry("""
nocom
noreorient
@He %f 0.0 0.00
@H  0 2.5 0
@H  0 -2.5 0
He -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Full_Molec.set_name("He2")

#Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 110,
    # 'DFT_RADIAL_POINTS':    5,
    'REFERENCE' : 'UKS'
})

# psi4.set_options({'cubeprop_tasks' : ['density'],
#                  'cubic_grid_spacing': [0.1, 0.1, 0.1]})

# energy_3, wfn_3 = psi4.energy("SVWN/cc-pvdz", molecule=mol_geometry, return_wfn=True)

#Make fragment calculations:
mol = pdft.U_Molecule(Full_Molec, "cc-pvdz", "SVWN")
f1  = pdft.U_Molecule(Monomer_2,  "cc-pvdz", "SVWN", jk=mol.jk)
f2  = pdft.U_Molecule(Monomer_1,  "cc-pvdz", "SVWN", jk=mol.jk)

#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)

pdfter.initial_run(1000)
dd = mol.to_grid(pdfter.fragments_Da + pdfter.fragments_Db - mol.Da.np - mol.Db.np)
pdft.plot1d_x(-dd, mol.Vpot, title="dd bond w/o vp", fignum=2, dimmer_length=bondlength)

rho_conv, ep_conv = pdfter.find_vp_response(maxiter=10, beta=4, atol=1e-5)
#%%
# pdfter.get_energies()
#%%
#%% Plotting Cubic
# #Set the box lenght and grid fineness.
# L = [8.0,  8.0, 8.0]
# D = [0.2, 0.2, 0.2]
# # Plot file
# O, N =  libcubeprop.build_grid(mol.wfn, L, D)
# block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
# f, (ax1, ax2) = plt.subplots(1, 2)
# libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, vp, name="He2", write_file=True)

#%%
vp_grid = mol.to_grid(pdfter.vp[0].np)
pdft.plot1d_x(vp_grid, mol.Vpot, title="vp bond:" + str(bondlength), fignum=4, dimmer_length=bondlength)
pdfter.get_density_sum()
dd = mol.to_grid(pdfter.fragments_Da + pdfter.fragments_Db - mol.Da.np - mol.Db.np)
pdft.plot1d_x(-dd, mol.Vpot, title="dd bond:" + str(bondlength), fignum=3, dimmer_length=bondlength)
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
