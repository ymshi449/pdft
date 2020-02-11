import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop

psi4.core.set_output_file("formic")
functional = 'svwn'
basis = 'sto-3g'

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

Full_Molec.set_name("Large")

#Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 434,
    # 'DFT_RADIAL_POINTS': 99,
    'REFERENCE' : 'UKS'})

#Make fragment calculations:
f1  = pdft.U_Molecule(Monomer_2,  basis, functional)
f2  = pdft.U_Molecule(Monomer_1,  basis, functional)
mol = pdft.U_Molecule(Full_Molec, basis, functional)


#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
pdfter.find_vp_response2(maxiter=25, beta=0.1, svd_rcond=1e-4)
# #%%
# # pdfter.get_energies()
# #%%
# # vp_plot = Cube(mol.wfn)
# #%%
# # vp_plot.plot_matrix(vp, 2,60)
# fig1 = plt.figure(num=None, figsize=(16, 12), dpi=160)
# plt.plot(rho_conv, figure=fig1)
# plt.xlabel(r"iteration")
# plt.ylabel(r"$\int |\rho_{whole} - \sum_{fragment} \rho|$")
# plt.title(r"Large Molecule (48 electrons) w/ density difference method ")
# # fig1.savefig("tests/rho")
# fig2 = plt.figure(num=None, figsize=(16, 12), dpi=160)
# plt.plot(ep_conv, figure=fig2)
# plt.xlabel(r"iteration")
# plt.ylabel(r"Ep")
# plt.title(r"Large w/ density difference method ")
# # fig2.savefig("tests/Ep")
#
# #%%
vp_psi4 = psi4.core.Matrix.from_array(pdfter.vp[0])
L = [4.0, 4.0, 4.0]
D = [0.05, 0.2, 0.2]
# Plot file
O, N = libcubeprop.build_grid(mol.wfn, L, D)
block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
vp_cube = libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, vp_psi4)
f, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=160)
p = ax.imshow(vp_cube[81, :, :], interpolation="bicubic")
ax.set_title("vp svd_rond=1e-5" + basis + functional)
f.colorbar(p, ax=ax)
f.show()