import psi4
import pdft
import matplotlib.pyplot as plt
import numpy as np
import libcubeprop

psi4.set_output_file("B2")
separation = 4.522
functional = 'svwn'
basis = 'STO-3G'

psi4.set_output_file("projection.psi4")

# Full_Molec = psi4.geometry("""
# nocom
# noreorient
# He %f 0.0 0.00
# He -%f 0.0 0.00
# units bohr
# symmetry c1
# """ % (separation / 2, separation / 2))
#
# Monomer_1 =  psi4.geometry("""
# nocom
# noreorient
# He %f 0.0 0.00
# @He -%f 0.0 0.00
# units bohr
# symmetry c1
# """ % (separation / 2, separation / 2))
#
# Monomer_2 =  psi4.geometry("""
# nocom
# noreorient
# @He %f 0.0 0.00
# He -%f 0.0 0.00
# units bohr
# symmetry c1
# """ % (separation / 2, separation / 2))
#
# # Make fragment calculations:
# f1  = pdft.U_Molecule(Monomer_2,  basis, "SVWN")
# f2  = pdft.U_Molecule(Monomer_1,  basis, "SVWN")
# mol = pdft.U_Molecule(Full_Molec, basis, "SVWN")

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

# Make fragment calculations:
f1  = pdft.U_Molecule(Monomer_2,  basis, "SVWN")
f2  = pdft.U_Molecule(Monomer_1,  basis, "SVWN")
mol = pdft.U_Molecule(Full_Molec, basis, "SVWN")

# Monomer_1 =  psi4.geometry("""
# nocom
# noreorient
# @H  -1 0 0
# H  1 0 0
# units bohr
# symmetry c1
# """)
#
# Monomer_2 =  psi4.geometry("""
# nocom
# noreorient
# H  -1 0 0
# @H  1 0 0
# units bohr
# symmetry c1
# """)
# Full_Molec =  psi4.geometry("""
# nocom
# noreorient
# 1 2
# H  -1 0 0
# H  1 0 0
# units bohr
# symmetry c1""")
# Full_Molec.set_name("H2P")
#
# Full_Molec.set_name("projection")
#
# #Psi4 Options:
# psi4.set_options({
#     # 'DFT_SPHERICAL_POINTS': 110,
#     # 'DFT_RADIAL_POINTS':    5,
#     'REFERENCE' : 'UKS'
# })
#
# #Make fragment calculations:
# f1  = pdft.U_Molecule(Monomer_2,  basis, "SVWN", omega=0.5)
# f2  = pdft.U_Molecule(Monomer_1,  basis, "SVWN", omega=0.5)
# mol = pdft.U_Molecule(Full_Molec, basis, "SVWN")

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
print("dn", np.sum(np.abs(rho_fragment_be - rho_molecule) * w))
print("Orthogonality", ortho)
print("eigens", f1.eig_a.np, f2.eig_a.np)

pdfter.fragments_scf(100)
pdfter.sym_Lowdin_ortho_4_MO()
D1a = np.copy(f1.Da.np)
D2a = np.copy(f2.Da.np)
D1b = np.copy(f1.Db.np)
D2b = np.copy(f2.Db.np)
drho_old = 0.0

for i in range(100):
    projection_method = "Frozen"
    P1a = np.copy(f2.Cocca.np)
    P1b = np.copy(f2.Coccb.np)
    P2a = np.copy(f1.Cocca.np)
    P2b = np.copy(f1.Coccb.np)
    mol.scf(100, projection=[P1a, P1b, projection_method])
    f1.Cocca.np[:] = np.copy(mol.Cocca.np[:,:f1.nalpha])
    f1.Coccb.np[:] = np.copy(mol.Coccb.np[:,:f1.nbeta])
    mol.scf(100, projection=[P2a, P2b, projection_method])
    f2.Cocca.np[:] = np.copy(mol.Cocca.np[:,:f2.nalpha])
    f2.Coccb.np[:] = np.copy(mol.Coccb.np[:,:f2.nbeta])
    pdfter.sym_Lowdin_ortho_4_MO()
    D1a = np.copy(f1.Da.np)
    D2a = np.copy(f2.Da.np)
    D1b = np.copy(f1.Db.np)
    D2b = np.copy(f2.Db.np)
    n1 = f1.omega*mol.to_grid(D1a+D1b)
    n2 = f2.omega*mol.to_grid(D2a+D2b)
    drho = np.sum(np.abs(n_mol - n1 - n2)*w)
    print(i, drho)
    # if np.isclose(drho_old, drho):
    #     break
    drho_old = drho

f,ax = plt.subplots(1,1, dpi=210)
pdft.plot1d_x(n_mol, mol.Vpot, ax=ax, label="n_mol", dimmer_length=separation, ls='-')
pdft.plot1d_x(n1+n2, mol.Vpot, ax=ax, label="nf", ls='-')
pdft.plot1d_x(n1, mol.Vpot, ax=ax, label="n1af", ls='--')
pdft.plot1d_x(n2, mol.Vpot, ax=ax, label="n2af", ls='--')
pdft.plot1d_x(n_mol - (n1+n2), mol.Vpot, ax=ax, label="dn", ls="dotted")
ax.legend()
f.show()
plt.close(f)