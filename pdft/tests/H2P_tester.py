import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop

Monomer_1 =  psi4.geometry("""
nocom
noreorient
@H  -1 0 0
@H  0 0 0
H  1 0 0
units bohr
symmetry c1
""")

Monomer_2 =  psi4.geometry("""
nocom
noreorient
H  -1 0 0
@H  0 0 0
@H  1 0 0
units bohr
symmetry c1
""")
Full_Molec =  psi4.geometry("""
nocom
noreorient
H  -1 0 0
@H  0 0 0
H  1 0 0
units bohr
symmetry c1""")
Full_Molec.set_name("H2P")

#Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 110,
    # 'DFT_RADIAL_POINTS':    5,
    'REFERENCE' : 'UKS'
})

# psi4.set_options({'cubeprop_tasks' : ['density'],
#                  'cubic_grid_spacing': [0.1, 0.1, 0.1]})

# energy_3, wfn_3 = psi4.energy("SVWN/cc-pVDZ", molecule=mol_geometry, return_wfn=True)

#Make fragment calculations:
f1  = pdft.U_Molecule(Monomer_2,  "6-311G", "SVWN")
f2  = pdft.U_Molecule(Monomer_1,  "6-311G", "SVWN")
mol = pdft.U_Molecule(Full_Molec, "6-311G", "SVWN")

#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
vp,vpa,vpb,rho_conv, ep_conv = pdfter.find_vp_response(maxiter=10, beta=1, atol=1e-5)
#%%
# pdfter.get_energies()
#%%
#%% Plotting
#Set the box lenght and grid fineness.
L = [8.0,  8.0, 8.0]
D = [0.2, 0.2, 0.2]
#%%
# Plot file
O, N =  libcubeprop.build_grid(mol.wfn, L, D)
block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
f, (ax1, ax2) = plt.subplots(1, 2)
vp_cube = libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, vp)
libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, vp, name="Large_vp", weite_file=True)
vp_cube1, _ = libcubeprop.cube_to_array("Large_vp.cube")

# #%%
# # vp_plot.plot_matrix(vp,2,60)
# vp_grid = mol.to_grid(vp.np)
# fig3 = plt.figure(num=3, figsize=(16, 12), dpi=160)
# pdft.plot1d_x(vp_grid, mol.Vpot, title="vp", figure=fig3)
#
# fig1 = plt.figure(num=1, figsize=(16, 12), dpi=160)
# plt.plot(rho_conv, figure=fig1)
# plt.xlabel(r"iteration")
# plt.ylabel(r"$\int |\rho_{whole} - \sum_{fragment} \rho|$")
# plt.title(r"$H2^+$ w/ response method ")
# fig1.savefig("rho")
# plt.show()
# fig2 = plt.figure(num=2, figsize=(16, 12), dpi=160)
# plt.plot(ep_conv, figure=fig2)
# plt.xlabel(r"iteration")
# plt.ylabel(r"Ep")
# plt.title(r"$H2^+$ w/ response method ")
# fig2.savefig("Ep")
# plt.show()