import psi4
import pdft
import matplotlib.pyplot as plt
import numpy as np
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
n1be = mol.to_grid(D1a + D1b)
n2be = mol.to_grid(D2a + D2b)
n_mol = mol.to_grid(mol.Da.np + mol.Db.np)
_, _, _, w = pdfter.molecule.Vpot.get_np_xyzw()
rho_molecule = mol.to_grid(mol.Da.np, Duv_b=mol.Db.np)
rho_fragment = mol.to_grid(pdfter.fragments_Da, Duv_b=pdfter.fragments_Db)

print(np.trace(np.dot(mol.Da.np + mol.Db.np - pdfter.fragments_Da - pdfter.fragments_Db, mol.T.np)))
print("dn", np.sum(np.abs(rho_fragment - rho_molecule) * w))
print("Orthogonality", np.trace(np.dot(f2.Da.np + f2.Db.np, S).dot(np.dot(f1.Da.np + f1.Db.np, S))))
# for i in range(1,100):
#     print("==============================================================")
#     if i < mol.eig_a.np.shape[0]:
#         mu = np.sort(np.abs(mol.eig_a.np))[i]
#     else:
#         mu *= 0.9
mu = np.min(np.abs(mol.eig_a.np))
step = np.exp(np.log(1e6/mu)/100.0)
i = 0
while True:
    print("===========================i=%i,mu=%e=================================="%(i,mu))
    mu *= step
    # if i < mol.eig_a.np.shape[0]:
    #     mu = np.sort(np.abs(mol.eig_a.np))[i]
    # else:
    #     mu *= 1.05
    P1 = np.dot(S, np.dot(f2.Da.np + f2.Db.np, S))
    P2 = np.dot(S, np.dot(f1.Da.np + f1.Db.np, S))
    P1psi = psi4.core.Matrix.from_array(mu*P1)
    P2psi = psi4.core.Matrix.from_array(mu*P2)
    f1.scf(maxiter=1000, vp_matrix=[P1psi, P1psi])
    f2.scf(maxiter=1000, vp_matrix=[P2psi, P2psi])
    pdfter.get_density_sum()
    rho_fragment = mol.to_grid(pdfter.fragments_Da, Duv_b=pdfter.fragments_Db)

    print("NAKP", np.trace(np.dot(mol.Da.np + mol.Db.np
                                     - pdfter.fragments_Da - pdfter.fragments_Db, mol.T.np)))
    print("dn", np.sum(np.abs(rho_fragment - rho_molecule) * w))
    print("Orthogonality", np.trace(np.dot(f2.Da.np + f2.Db.np, S).dot(np.dot(f1.Da.np + f1.Db.np, S))))
    i += 1
    if i >= 100:
        break
    # if abs(f1.frag_energy - E1_old) < 1e-4:
    #     break

n1af = mol.to_grid(f1.Da.np + f1.Db.np)
n2af = mol.to_grid(f2.Da.np + f2.Db.np)

f,ax = plt.subplots(1,1, dpi=210)
pdft.plot1d_x(n1af, mol.Vpot, ax=ax, label="n1af", dimmer_length=separation, ls='--')
pdft.plot1d_x(n2af, mol.Vpot, ax=ax, label="n2af", title="%e"%mu, ls='--')
pdft.plot1d_x(n1be, mol.Vpot, ax=ax, label="n1be", ls="dotted")
pdft.plot1d_x(n2be, mol.Vpot, ax=ax, label="n2be", ls="dotted")
ax.legend()
f.show()
f.savefig("%1.0e"%mu)
plt.close(f)