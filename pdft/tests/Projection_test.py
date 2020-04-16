import psi4
import pdft
import matplotlib.pyplot as plt
import numpy as np
import libcubeprop

psi4.set_output_file("B2")
separation = 4.522
functional = 'svwn'
basis = 'cc-pvtz'

psi4.set_output_file("projection.psi4")

# Full_Molec = psi4.geometry("""
# nocom
# noreorient
# Be %f 0.0 0.00
# Be -%f 0.0 0.00
# units bohr
# symmetry c1
# """ % (separation / 2, separation / 2))
#
# Monomer_1 =  psi4.geometry("""
# nocom
# noreorient
# Be %f 0.0 0.00
# @Be -%f 0.0 0.00
# units bohr
# symmetry c1
# """ % (separation / 2, separation / 2))
#
# Monomer_2 =  psi4.geometry("""
# nocom
# noreorient
# @Be %f 0.0 0.00
# Be -%f 0.0 0.00
# units bohr
# symmetry c1
# """ % (separation / 2, separation / 2))
#
# # Make fragment calculations:
# f1  = pdft.U_Molecule(Monomer_2,  basis, "SVWN")
# f2  = pdft.U_Molecule(Monomer_1,  basis, "SVWN")
# mol = pdft.U_Molecule(Full_Molec, basis, "SVWN")

Monomer_1 =  psi4.geometry("""
nocom
noreorient
@H  -1 0 0
H  1 0 0
units bohr
symmetry c1
""")

Monomer_2 =  psi4.geometry("""
nocom
noreorient
H  -1 0 0
@H  1 0 0
units bohr
symmetry c1
""")
Full_Molec =  psi4.geometry("""
nocom
noreorient
1 2
H  -1 0 0
H  1 0 0
units bohr
symmetry c1""")
Full_Molec.set_name("H2P")

Full_Molec.set_name("projection")

#Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 110,
    # 'DFT_RADIAL_POINTS':    5,
    'REFERENCE' : 'UKS'
})

#Make fragment calculations:
f1  = pdft.U_Molecule(Monomer_2,  basis, "SVWN", omega=0.5)
f2  = pdft.U_Molecule(Monomer_1,  basis, "SVWN", omega=0.5)
mol = pdft.U_Molecule(Full_Molec, basis, "SVWN")

#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
print("-----------Isolated systems-------------")
pdfter.fragments_scf(100)
D1a = np.copy(f1.Da.np)
D2a = np.copy(f2.Da.np)
D1b = np.copy(f1.Db.np)
D2b = np.copy(f2.Db.np)
C1a = np.copy(f1.Ca.np)
C2a = np.copy(f2.Ca.np)
C1b = np.copy(f1.Cb.np)
C2b = np.copy(f2.Cb.np)
E1 = f1.energy
E2 = f2.energy
S = np.array(mol.mints.ao_overlap())
n1be = mol.to_grid(D1a + D1b)
n2be = mol.to_grid(D2a + D2b)
n_mol = mol.to_grid(mol.Da.np + mol.Db.np)
_, _, _, w = pdfter.molecule.Vpot.get_np_xyzw()
rho_molecule = mol.to_grid(mol.Da.np, Duv_b=mol.Db.np)
rho_fragment_be = mol.to_grid(pdfter.fragments_Da, Duv_b=pdfter.fragments_Db)
ortho = [np.dot(f1.Ca.np.T,
                S.dot(C2a[:, 0:f2.nalpha])),
         np.dot(f1.Cb.np.T,
                S.dot(C2b[:, 0:f2.nbeta])),
         np.dot(f2.Ca.np.T,
                S.dot(C1a[:, 0:f1.nalpha])),
         np.dot(f2.Cb.np.T,
                S.dot(C1b[:, 0:f1.nbeta])),
         np.dot(mol.Ca.np.T,
                S.dot(C1a[:, 0:f2.nalpha])),
         np.dot(mol.Cb.np.T,
                S.dot(C1b[:, 0:f2.nbeta]))
         ]
print("dn", np.sum(np.abs(rho_fragment_be - rho_molecule) * w))
print("Orthogonality", ortho)
print("eigens", f1.eig_a.np, f2.eig_a.np)

projection_method = "Huzinaga"
P1a = np.dot(f2.Da.np, S)
P1b = np.dot(f2.Db.np, S)
P2a = np.dot(f1.Da.np, S)
P2b = np.dot(f1.Db.np, S)
# P1a = S.dot(P1a)
# P1b = S.dot(P1b)
# P2a = S.dot(P2a)
# P2b = S.dot(P2b)
# print("--------------")
# f1.scf(maxiter=100, projection=[P1a, P1b, projection_method], print_energies=True, mu=1e7)
# print("--------------")
# f2.scf(maxiter=100, projection=[P2a, P2b, projection_method], print_energies=True, mu=1e7)
# print("--------------")
# mol.scf(maxiter=100, projection=[P2a, P2b, projection_method], print_energies=True, mu=1e7)
print("----------Orthogonalization---------")
pdfter.orthogonal_scf(35, projection_method=projection_method, mixing_paramter=1, mu=1e7, printflag=False)
# print("--------------")
ortho = [np.dot(f1.Cocca.np.T,
                S.dot(C2a[:, 0:f2.nalpha])),
         np.dot(f1.Coccb.np.T,
                S.dot(C2b[:, 0:f2.nbeta])),
         np.dot(f2.Cocca.np.T,
                S.dot(C1a[:, 0:f1.nalpha])),
         np.dot(f2.Coccb.np.T,
                S.dot(C1b[:, 0:f1.nbeta])),
         np.dot(f1.Cocca.np.T,
                S.dot(f2.Cocca.np)),
         np.dot(f1.Coccb.np.T,
                S.dot(f2.Coccb.np))
         ]
rho_fragment = mol.to_grid(pdfter.fragments_Da, Duv_b=pdfter.fragments_Db)
print("dn", np.sum(np.abs(rho_fragment - rho_molecule) * w))
print("before-after", np.sum(np.abs(rho_fragment - rho_fragment_be) * w))
print("Orthogonality", ortho)

n1af = mol.to_grid(f1.Da.np + f1.Db.np)
n2af = mol.to_grid(f2.Da.np + f2.Db.np)

f,ax = plt.subplots(1,1, dpi=210)
pdft.plot1d_x(n_mol, mol.Vpot, ax=ax, label="n_mol", dimmer_length=separation, ls='-')
pdft.plot1d_x(0.5*n1af, mol.Vpot, ax=ax, label="n1af", dimmer_length=separation, ls='-')
pdft.plot1d_x(0.5*n2af, mol.Vpot, ax=ax, label="n2af", ls='-')
pdft.plot1d_x(0.5*n1be, mol.Vpot, ax=ax, label="n1be", ls="--")
pdft.plot1d_x(0.5*n2be, mol.Vpot, ax=ax, label="n2be", ls="--")
pdft.plot1d_x(n_mol - 0.5*(n1af+n2af), mol.Vpot, ax=ax, label="dnaf", ls="dotted")
pdft.plot1d_x(n_mol - 0.5*(n1be+n2be), mol.Vpot, ax=ax, label="dnbe", ls="dotted")
ax.legend()
f.show()
plt.close(f)

# pdfter.get_density_sum()
# # #%% 1 basis 2D plot
# L = [4.0, 4.0, 2.0]
# D = [0.1, 0.1, 0.1]
# # Plot file
# O, N = libcubeprop.build_grid(mol.wfn, L, D)
# block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
# D_mol = psi4.core.Matrix.from_array(D1a+D1b+D2a+D2b)
# D_f = psi4.core.Matrix.from_array(pdfter.fragments_Da+pdfter.fragments_Db)
# n_mol_cube = libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, D_mol)
# n_f_cube = libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, D_f)
#
# f, ax = plt.subplots(1, 1, dpi=160)
# p = ax.imshow(n_mol_cube[:, :, 20], interpolation="bicubic", cmap="Spectral")
# atoms = libcubeprop.get_atoms(mol.wfn, D, O)
# ax.scatter(atoms[:,2], atoms[:,1])
# ax.set_title("nbe")
# f.colorbar(p, ax=ax)
# f.show()
# f.savefig("vp")
#
# f, ax = plt.subplots(1, 1, dpi=160)
# p = ax.imshow(n_f_cube[:, :, 20], interpolation="bicubic", cmap="Spectral")
# atoms = libcubeprop.get_atoms(mol.wfn, D, O)
# ax.scatter(atoms[:,2], atoms[:,1])
# ax.set_title("naf")
# f.colorbar(p, ax=ax)
# f.show()
# f.savefig("vp")
#
# f, ax = plt.subplots(1, 1, dpi=160)
# p = ax.imshow((n_mol_cube-n_f_cube)[:, :, 20], interpolation="bicubic", cmap="Spectral")
# atoms = libcubeprop.get_atoms(mol.wfn, D, O)
# ax.scatter(atoms[:,2], atoms[:,1])
# ax.set_title("dn")
# f.colorbar(p, ax=ax)
# f.show()
