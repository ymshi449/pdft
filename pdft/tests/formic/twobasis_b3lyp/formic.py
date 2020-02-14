import psi4
import numpy as np
import pdft
import matplotlib.pyplot as plt
import libcubeprop
import pickle

psi4.core.set_output_file("formic")
functional = 'b3lyp'
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

Full_Molec.set_name("formic")

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
#3
pdfter.find_vp_response2(maxiter=77, beta=0.1, svd_rcond=1e-3)

#data = {"drho": pdfter.drho_conv, "Ep": pdfter.ep_conv,"vp": pdfter.vp[0]}
#pickle.dump(data, open( "save.p", "wb" ), protocol=4)

pdfter.ep_conv = np.array(pdfter.ep_conv)
f,ax = plt.subplots(1,1)
ax.plot(np.log10(np.abs(pdfter.ep_conv[1:] - pdfter.ep_conv[:-1])), "o")
ax.set_title("log dEp")
f.savefig("dEp3")

#%% 2D Plot file
L = [2.0,  2.0, 2.0]
D = [0.05, 0.1, 0.1]
vp_psi4 = psi4.core.Matrix.from_array(pdfter.vp[0])
O, N =  libcubeprop.build_grid(mol.wfn, L, D)
block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, vp_psi4, name="formic3", write_file=True)

#4
pdfter.find_vp_response2(maxiter=77, beta=0.1, svd_rcond=1e-4)

pdfter.ep_conv = np.array(pdfter.ep_conv)
f,ax = plt.subplots(1,1)
ax.plot(np.log10(np.abs(pdfter.ep_conv[1:] - pdfter.ep_conv[:-1])), "o")
ax.set_title("log dEp")
f.savefig("dEp4")

#%% 2D Plot file
vp_psi4 = psi4.core.Matrix.from_array(pdfter.vp[0])
libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, vp_psi4, name="formic4", write_file=True)

#5
pdfter.find_vp_response2(maxiter=77, beta=0.1, svd_rcond=1e-5)

pdfter.ep_conv = np.array(pdfter.ep_conv)
f,ax = plt.subplots(1,1)
ax.plot(np.log10(np.abs(pdfter.ep_conv[1:] - pdfter.ep_conv[:-1])), "o")
ax.set_title("log dEp")
f.savefig("dEp5")

#%% 2D Plot file
vp_psi4 = psi4.core.Matrix.from_array(pdfter.vp[0])
libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, vp_psi4, name="formic5", write_file=True)

#6
pdfter.find_vp_response2(maxiter=77, beta=0.1, svd_rcond=1e-6)

pdfter.ep_conv = np.array(pdfter.ep_conv)
f,ax = plt.subplots(1,1)
ax.plot(np.log10(np.abs(pdfter.ep_conv[1:] - pdfter.ep_conv[:-1])), "o")
ax.set_title("log dEp")
f.savefig("dEp6")

#%% 2D Plot file
vp_psi4 = psi4.core.Matrix.from_array(pdfter.vp[0])
libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, vp_psi4, name="formic6", write_file=True)

#7
pdfter.find_vp_response2(maxiter=77, beta=0.1, svd_rcond=1e-7)

pdfter.ep_conv = np.array(pdfter.ep_conv)
f,ax = plt.subplots(1,1)
ax.plot(np.log10(np.abs(pdfter.ep_conv[1:] - pdfter.ep_conv[:-1])), "o")
ax.set_title("log dEp")
f.savefig("dEp7")

#%% 2D Plot file
vp_psi4 = psi4.core.Matrix.from_array(pdfter.vp[0])
libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, vp_psi4, name="formic7", write_file=True)
