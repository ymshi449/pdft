import psi4
import pdft
import libcubeprop
import matplotlib.pyplot as plt

Full_Molec = psi4.geometry("""
nocom
noreorient
C    0.0000000    0.1929272   -1.9035340
O    0.0000000    1.1595219   -1.1616236
O    0.0000000   -1.0680669   -1.5349870
H    0.0000000    0.2949802   -2.9949776
H    0.0000000   -1.1409414   -0.5399614
C    0.0000000   -0.1929272    1.9035340
O    0.0000000   -1.1595219    1.1616236
O    0.0000000    1.0680669    1.5349870
H    0.0000000   -0.2949802    2.9949776
H    0.0000000    1.1409414    0.5399614
units bohr
symmetry c1
""")

Monomer_1 = psi4.geometry("""
nocom
noreorient
@C    0.0000000    0.1929272   -1.9035340
@O    0.0000000    1.1595219   -1.1616236
@O    0.0000000   -1.0680669   -1.5349870
@H    0.0000000    0.2949802   -2.9949776
@H    0.0000000   -1.1409414   -0.5399614
C    0.0000000   -0.1929272    1.9035340
O    0.0000000   -1.1595219    1.1616236
O    0.0000000    1.0680669    1.5349870
H    0.0000000   -0.2949802    2.9949776
H    0.0000000    1.1409414    0.5399614
units bohr
symmetry c1
""")

Monomer_2 = psi4.geometry("""
nocom
noreorient
C    0.0000000    0.1929272   -1.9035340
O    0.0000000    1.1595219   -1.1616236
O    0.0000000   -1.0680669   -1.5349870
H    0.0000000    0.2949802   -2.9949776
H    0.0000000   -1.1409414   -0.5399614
@C    0.0000000   -0.1929272    1.9035340
@O    0.0000000   -1.1595219    1.1616236
@O    0.0000000    1.0680669    1.5349870
@H    0.0000000   -0.2949802    2.9949776
@H    0.0000000    1.1409414    0.5399614
units bohr
symmetry c1
""")

Full_Molec.set_name("formic")

#Psi4 Options:
psi4.set_options({'DFT_SPHERICAL_POINTS': 434,
                  'DFT_RADIAL_POINTS': 99,
                  'REFERENCE' : 'UKS'})

#Make fragment calculations:
f1  = pdft.U_Molecule(Monomer_2,  "cc-pvdz", "SVWN")
f2  = pdft.U_Molecule(Monomer_1,  "cc-pvdz", "SVWN")
mol = pdft.U_Molecule(Full_Molec, "cc-pvdz", "SVWN")

#%%Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
vp,vpa,vpb,rho_conv, ep_conv = pdfter.find_vp_response(maxiter=100, beta=5, atol=1e-5)

#%% Plotting
# #Set the box lenght and grid fineness.
# L = [8.0,  8.0, 8.0]
# D = [0.2, 0.2, 0.2]
# #%%
# # Plot points
# O, N =  libcubeprop.build_grid(mol.wfn, L, D)
# block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12), dpi=160)
# # # Plot vp
# vp_cube = libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, vp, "Large_vp")
# # vp_cube, cube_info = libcubeprop.cube_to_array("Large_vp.cube")
# pt1 = ax1.imshow(vp_cube[40, :, :], interpolation="bicubic")
# fig.colorbar(pt1, ax=ax1)
# ax1.set_title("vp")
# # Plot density
# density = psi4.core.Matrix.from_array(mol.Da.np + mol.Db.np)
# rho_cube = libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, density, "Large_rho")
# del block
# del points
# del nxyz
# del npoints
# # rho_cube, cube_info = libcubeprop.cube_to_array("Large_rho.cube")
# pt2 = ax2.imshow(rho_cube[40, :, :], interpolation="bicubic")
# fig.colorbar(pt2, ax=ax2)
# ax2.set_title("Density")
# ## max index
# # np.unravel_index(np.argmax(np.abs(h2o_cube), axis=None), np.abs(h2o_cube).shape)

# #%%
# fig1 = plt.figure(num=None, figsize=(16, 12), dpi=160)
# plt.plot(rho_conv, figure=fig1)
# plt.xlabel(r"iteration")
# plt.ylabel(r"$\int |\rho_{whole} - \sum_{fragment} \rho|$")
# plt.title(r"Large Molecule (48 electrons) w/ density difference method ")
# fig1.savefig("rho")
# fig2 = plt.figure(num=None, figsize=(16, 12), dpi=160)
# plt.plot(ep_conv, figure=fig2)
# plt.xlabel(r"iteration")
# plt.ylabel(r"Ep")
# plt.title(r"Large w/ density difference method ")
# fig2.savefig("Ep")