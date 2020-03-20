import psi4
import pdft
import matplotlib.pyplot as plt
import numpy as np
import libcubeprop

psi4.set_output_file("B2")
separation = 4.522
functional = 'svwn'
basis = 'cc-pvdz'

psi4.set_output_file("projection.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
Be %f 0.0 0.00
Be -%f 0.0 0.00
units bohr
symmetry c1
""" % (separation / 2, separation / 2))

Monomer_1 =  psi4.geometry("""
nocom
noreorient
Be %f 0.0 0.00
@Be -%f 0.0 0.00
units bohr
symmetry c1
""" % (separation / 2, separation / 2))

Monomer_2 =  psi4.geometry("""
nocom
noreorient
@Be %f 0.0 0.00
Be -%f 0.0 0.00
units bohr
symmetry c1
""" % (separation / 2, separation / 2))

Full_Molec.set_name("projection")

#Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 110,
    # 'DFT_RADIAL_POINTS':    5,
    'REFERENCE' : 'UKS'
})

#Make fragment calculations:
f1  = pdft.U_Molecule(Monomer_2,  "cc-pvdz", "SVWN")
f2  = pdft.U_Molecule(Monomer_1,  "cc-pvdz", "SVWN")
mol = pdft.U_Molecule(Full_Molec, "CC-PVDZ", "SVWN")

#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)

pdfter.fragments_scf(100)
D1a = f1.Da.np
D2a = f2.Da.np
D1b = f1.Db.np
D2b = f2.Db.np
C1a = f1.Ca.np
C2a = f2.Ca.np
C1b = f1.Cb.np
C2b = f2.Cb.np
E1 = f1.energy
E2 = f2.energy
S = np.array(mol.mints.ao_overlap())
F = mol.Fa.np+mol.Fb.np
n1be = mol.to_grid(D1a + D1b)
n2be = mol.to_grid(D2a + D2b)
n_mol = mol.to_grid(mol.Da.np + mol.Db.np)
_, _, _, w = pdfter.molecule.Vpot.get_np_xyzw()
rho_molecule = mol.to_grid(mol.Da.np, Duv_b=mol.Db.np)
rho_fragment = mol.to_grid(pdfter.fragments_Da, Duv_b=pdfter.fragments_Db)
orho = [np.trace(np.dot(f2.Da.np, S).dot(np.dot(f1.Da.np, S))),
        np.trace(np.dot(f2.Db.np, S).dot(np.dot(f1.Db.np, S))),
        np.trace(np.dot(f2.Da.np, S).dot(np.dot(f1.Db.np, S))),
        np.trace(np.dot(f2.Db.np, S).dot(np.dot(f1.Da.np, S)))]
print(np.trace(np.dot(mol.Da.np + mol.Db.np - pdfter.fragments_Da - pdfter.fragments_Db, mol.T.np)))
print("dn", np.sum(np.abs(rho_fragment - rho_molecule) * w))
print("Orthogonality", orho)
print(f1.Ca.np.T.dot(S.dot(C2a[:,0])))
print(f2.Ca.np.T.dot(S.dot(C1a[:,0])))
print("eigens", f1.eig_a.np, f2.eig_a.np)
# for i in range(1,100):
#     print("==============================================================")
#     if i < mol.eig_a.np.shape[0]:
#         mu = np.sort(np.abs(mol.eig_a.np))[i]
#     else:
#         mu *= 0.9
mu = np.min(np.abs(mol.eig_a.np))
# step = np.exp(np.log(1e6/mu)/100.0)
# i = 0
# while True:
#     print("===========================i=%i,mu=%e=================================="%(i,mu))
#     mu *= step
#     P1a = np.dot(f2.Da.np, S)
#     P1b = np.dot(f2.Db.np, S)
#     P2a = np.dot(f1.Da.np, S)
#     P2b = np.dot(f1.Db.np, S)
#     # P1 = -0.5*(np.dot(F, np.dot(f2.Da.np + f2.Db.np, S)) +
#     #            np.dot(S, np.dot(f2.Da.np + f2.Db.np, F)))
#     # P2 = -0.5*(np.dot(F, np.dot(f1.Da.np + f1.Db.np, S)) +
#     #            np.dot(S, np.dot(f1.Da.np + f1.Db.np, F)))
#     P1psi = psi4.core.Matrix.from_array([P1a, P1b])
#     P2psi = psi4.core.Matrix.from_array([P2a, P2b])
#     f1.scf(maxiter=1000, projection=[P1a, P1b])
#     f2.scf(maxiter=1000, projection=[P2a, P2b])
#     pdfter.get_density_sum()
#     rho_fragment = mol.to_grid(pdfter.fragments_Da, Duv_b=pdfter.fragments_Db)
#
#     print("NAKP", np.trace(np.dot(mol.Da.np + mol.Db.np
#                                      - pdfter.fragments_Da - pdfter.fragments_Db, mol.T.np)))
#     print("dn", np.sum(np.abs(rho_fragment - rho_molecule) * w))
#     print("Orthogonality", np.trace(np.dot(f2.Da.np + f2.Db.np, S).dot(np.dot(f1.Da.np + f1.Db.np, S))))
#     i += 1
#     if i >= 3:
#         break

P1a = np.dot(f2.Da.np, S)
P1b = np.dot(f2.Db.np, S)
P2a = np.dot(f1.Da.np, S)
P2b = np.dot(f1.Db.np, S)
# P1 = -0.5*(np.dot(F, np.dot(f2.Da.np + f2.Db.np, S)) +
#            np.dot(S, np.dot(f2.Da.np + f2.Db.np, F)))
# P2 = -0.5*(np.dot(F, np.dot(f1.Da.np + f1.Db.np, S)) +
#            np.dot(S, np.dot(f1.Da.np + f1.Db.np, F)))
P1psi = psi4.core.Matrix.from_array([P1a, P1b])
P2psi = psi4.core.Matrix.from_array([P2a, P2b])
f1.scf(maxiter=1000, projection=[P1a, P1b], print_energies=True)
print("--------------")
f2.scf(maxiter=1000, projection=[P2a, P2b], print_energies=True)
pdfter.get_density_sum()
rho_fragment = mol.to_grid(pdfter.fragments_Da, Duv_b=pdfter.fragments_Db)
orho = [np.trace(np.dot(f2.Da.np, S).dot(np.dot(f1.Da.np, S))),
        np.trace(np.dot(f2.Db.np, S).dot(np.dot(f1.Db.np, S))),
        np.trace(np.dot(f2.Da.np, S).dot(np.dot(f1.Db.np, S))),
        np.trace(np.dot(f2.Db.np, S).dot(np.dot(f1.Da.np, S)))]
print("NAKP", np.trace(np.dot(mol.Da.np + mol.Db.np
                                 - pdfter.fragments_Da - pdfter.fragments_Db, mol.T.np)))
print("dn", np.sum(np.abs(rho_fragment - rho_molecule) * w))
print("Orthogonality", orho)
print(f1.Ca.np.T.dot(S.dot(C2a[:,0])))
print(f2.Ca.np.T.dot(S.dot(C1a[:,0])))
print(f1.Ca.np.T.dot(S.dot(f2.Ca.np[:,0])))
print(f2.Ca.np.T.dot(S.dot(f1.Ca.np[:,0])))
print("eigens", f1.eig_a.np, f2.eig_a.np)

n1af = mol.to_grid(f1.Da.np + f1.Db.np)
n2af = mol.to_grid(f2.Da.np + f2.Db.np)

f,ax = plt.subplots(1,1, dpi=210)
pdft.plot1d_x(n1af, mol.Vpot, ax=ax, label="n1af", dimmer_length=separation, ls='--')
pdft.plot1d_x(n2af, mol.Vpot, ax=ax, label="n2af", title="%e"%mu, ls='--')
pdft.plot1d_x(n1be, mol.Vpot, ax=ax, label="n1be", ls="dotted")
pdft.plot1d_x(n2be, mol.Vpot, ax=ax, label="n2be", ls="dotted")
pdft.plot1d_x(n1af + n2af - n1be - n2be, mol.Vpot, ax=ax, label="dn", ls="dotted")
ax.legend()
f.show()
f.savefig("%1.0e"%mu)
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
