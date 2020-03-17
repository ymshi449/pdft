import psi4
import pdft
import matplotlib.pyplot as plt
import numpy as np
psi4.set_output_file("H2")
bondlength = 1.446

Full_Molec =  psi4.geometry( """
nocom
noreorient
H %f 0.0 0.00
H -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Monomer_1 =  psi4.geometry("""
nocom
noreorient
H %f 0.0 0.00
@H -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Monomer_2 =  psi4.geometry("""
nocom
noreorient
@H %f 0.0 0.00
H -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

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
# mu = 1e7

E1_old = E1
for i in range(1,700):
    mu = 10**(i/100.0)
    P1 = np.dot(S, np.dot(f2.Da.np + f2.Db.np, S))
    P2 = np.dot(S, np.dot(f1.Da.np + f1.Db.np, S))
    P1psi = psi4.core.Matrix.from_array(mu*P1)
    P2psi = psi4.core.Matrix.from_array(mu*P2)
    f1.scf(maxiter=1000, vp_matrix=[P1psi, P1psi])
    f2.scf(maxiter=1000, vp_matrix=[P2psi, P2psi])
    print(i, f1.frag_energy - E1_old)
    # if abs(f1.frag_energy - E1_old) < 1e-4:
    #     break
    E1_old = f1.frag_energy

n1af = mol.to_grid(f1.Da.np + f1.Db.np)
n2af = mol.to_grid(f2.Da.np + f2.Db.np)

f,ax = plt.subplots(1,1, dpi=210)
pdft.plot1d_x(n1af, mol.Vpot, ax=ax, label="n1af", dimmer_length=bondlength, ls='--')
pdft.plot1d_x(n2af, mol.Vpot, ax=ax, label="n2af", title="%e"%mu, ls='--')
pdft.plot1d_x(n1be, mol.Vpot, ax=ax, label="n1be", ls="dotted")
pdft.plot1d_x(n2be, mol.Vpot, ax=ax, label="n2be", ls="dotted")
ax.legend()
f.show()
f.savefig("%1.0e"%mu)
plt.close(f)