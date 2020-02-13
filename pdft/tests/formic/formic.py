import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop

psi4.core.set_output_file("formic.psi4")
functional = 'svwn'
basis = 'cc-pvdz'

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

# pdfter.find_vp_response(maxiter=25, beta=0.1, svd_rcond=1e-4)
pdfter.find_vp_response_1basis(21, svd_rcond=1e-4, regul_const=None, beta=0.1, a_rho_var=1e-7)

#%% 2 basis 2D plot
# vp_psi4 = psi4.core.Matrix.from_array(pdfter.vp[0])
# L = [4.0, 4.0, 4.0]
# D = [0.05, 0.2, 0.2]
# # Plot file
# O, N = libcubeprop.build_grid(mol.wfn, L, D)
# block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
# vp_cube = libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, vp_psi4)
# f, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=160)
# p = ax.imshow(vp_cube[81, :, :], interpolation="bicubic")
# ax.set_title("vp svd_rond=1e-5" + basis + functional)
# f.colorbar(p, ax=ax)
# f.show()

#%% 1 basis 2D plot
L = [4.0, 4.0, 4.0]
D = [0.05, 0.2, 0.2]
# Plot file
O, N = libcubeprop.build_grid(mol.wfn, L, D)
block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
vp_cube = libcubeprop.compute_density_1basis(mol.wfn, O, N, D, npoints, points, nxyz, block, pdfter.vp[0])
f, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=160)
p = ax.imshow(vp_cube[81, :, :], interpolation="bicubic", cmap="Spectral")
ax.set_title("vp svd_rond=1e-5" + basis + functional)
f.colorbar(p, ax=ax)
f.show()
f.savefig("formic_svd1e-5_1b")