import sys
sys.path.append('../')
import scipy.ndimage
import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np
import pickle

# Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 110,
    # 'DFT_RADIAL_POINTS':    5,
    'REFERENCE': 'UKS'
})
psi4.set_output_file("HeALL96")
bindingenergy = []
bindinglength = []
vp = []
mol_energy = []
f1_energy = []
f2_energy = []
ep = []
nuclear_nad = []
vp_svwn = []
for bondlength in np.linspace(4.6, 20, 35):
    print("============%f==============" % bondlength)
    Full_Molec =  psi4.geometry("""
    nocom
    noreorient
    He %f 0.0 0.00
    He -%f 0.0 0.00
    units bohr
    symmetry c1
    """ % (bondlength / 2, bondlength / 2))

    Monomer_1 =  psi4.geometry("""
    nocom
    noreorient
    He %f 0.0 0.00
    @He -%f 0.0 0.00
    units bohr
    symmetry c1
    """ % (bondlength / 2, bondlength / 2))

    Monomer_2 =  psi4.geometry("""
    nocom
    noreorient
    @He %f 0.0 0.00
    He -%f 0.0 0.00
    units bohr
    symmetry c1
    """ % (bondlength / 2, bondlength / 2))

    Full_Molec.set_name("He2")

    #Make fragment calculations:
    mol = pdft.U_Molecule(Full_Molec, "aug-cc-pvdz", "svwn")
    f1  = pdft.U_Molecule(Monomer_2,  "aug-cc-pvdz", "svwn", jk=mol.jk)
    f2  = pdft.U_Molecule(Monomer_1,  "aug-cc-pvdz", "svwn", jk=mol.jk)

    #Start a pdft system
    pdfter = pdft.U_Embedding([f1, f2], mol)

    # Run with vp = vp_all96
    energy, vp_all96, vp_fock_all96 = pdfter.find_vp_all96(100, 1000, rtol=1e-2, hard_cutoff=True)
    # From Hartree to Ry
    energy = energy * 2.0

    # Append info
    vp.append(vp_all96)
    bindingenergy.append(energy)
    bindinglength.append(bondlength)
    mol_energy.append(mol.frag_energy*2)
    f1_energy.append(f1.frag_energy*2)
    f2_energy.append(f2.frag_energy*2)
    ep.append(pdfter.ep_conv[-1]*2)
    nuclear_nad.append(mol.Enuc - f1.Enuc - f2.Enuc)
    vp_grid = mol.to_grid(pdfter.vp[0])
    vp_svwn.append(vp)

x_HeHeccpvdz = (np.array(bindinglength).astype(float))
y_HeHeccpvdz = np.array(bindingenergy)

data = {"x": x_HeHeccpvdz,
        "y": y_HeHeccpvdz,
        "vp": vp,
        "mol_energy": mol_energy,
        "f1_energy": f1_energy,
        "f2_energy": f2_energy,
        "ep": ep,
        "nuclear_nad": nuclear_nad,
        "vp_svwn": vp_svwn
        }
pickle.dump(data, open( "save.p", "wb" ))

C6 = np.polynomial.polynomial.polyfit(x_HeHeccpvdz**-1, -y_HeHeccpvdz,[6], w=x_HeHeccpvdz**6)[-1]
print("C6", C6)
fig1, ax1 = plt.subplots(1, 1, figsize=(16, 12), dpi=160)
ax1.plot(np.log(x_HeHeccpvdz), np.log(-y_HeHeccpvdz),'ro')
ax1.plot(np.log(np.linspace(x_HeHeccpvdz[0], x_HeHeccpvdz[-1],num=100)),
         -6*np.log(np.linspace(x_HeHeccpvdz[0], x_HeHeccpvdz[-1], num=100)) + np.log(C6))
fig1.savefig("C6 fitting")

fig2, ax2 = plt.subplots(1, 1, figsize=(16, 12), dpi=160)
ax2.scatter(x_HeHeccpvdz, f1_energy + f2_energy + ep + y_HeHeccpvdz + nuclear_nad)
fig2.savefig("Energy")


#%% Running SCF without any vp.
# pdfter.fragments_scf(max_iter=1000)
# n_novp = mol.to_grid(pdfter.fragments_Da + pdfter.fragments_Db)

#%% Running SCF with pbe
# rho_conv, ep_conv = pdfter.find_vp(maxiter=21, beta=1, atol=1e-5)
# pdfter.get_density_sum()
# n_pbe = mol.to_grid(pdfter.fragments_Da + pdfter.fragments_Db)
# pdft.plot1d_x(n_pbe - n_novp, mol.Vpot, title="density difference pbe - novp" + str(bondlength), fignum=0)
# vp_pbe = mol.to_grid((pdfter.vp[0] + pdfter.vp[1])*0.5)
# vp_fock_pbe = pdfter.vp[0]
# pdft.plot1d_x(vp_pbe, mol.Vpot, title="vp density_difference" + str(bondlength), fignum=1, dimmer_length=bondlength)

#%% Get vp_all96
# all96_e, vp_all96, vp_fock_all96 = pdfter.vp_all96()
# print("E all96", all96_e)
# pdft.plot1d_x(vp_all96, mol.Vpot, title="vp all96:" + str(bondlength), fignum=2)

# #%% Running SCF with vp_all96
# vp_fock_psi4 = psi4.core.Matrix.from_array(vp_fock_all96)
# pdfter.fragments_scf(max_iter=1000, vp=True, vp_fock=[vp_fock_psi4, vp_fock_psi4])
# n_all96 = mol.to_grid(pdfter.fragments_Da + pdfter.fragments_Db)
# pdft.plot1d_x(n_all96 - n_novp, mol.Vpot, title="density difference all96 - novp" + str(bondlength), fignum=3)
# pdft.plot1d_x(n_all96 - n_pbe, mol.Vpot, title="density difference all96 - pbe" + str(bondlength), fignum=4)
# #%% Campare density difference
# w = mol.Vpot.get_np_xyzw()[-1]
# print("==========DENSITY DIFFERENCE==========")
# print("Density difference no vp and all96.", np.sum(np.abs(n_novp - n_all96)*w))
# print("Density difference pbe and all96.", np.sum(np.abs(n_all96 - n_pbe)*w))
# print("Density difference no vp and pbe.", np.sum(np.abs(n_novp - n_pbe)*w))
#

# #%% Check if vp_all96 consist with vp_fock_all96
# if pdfter.four_overlap is None:
#     pdfter.four_overlap, _, _, _ = pdft.fouroverlap(pdfter.molecule.wfn, pdfter.molecule.geometry,
#                                                     pdfter.molecule.basis, pdfter.molecule.mints)

# vp_basis_all96 = mol.to_basis(vp_all96)

# vp_fock_all96_1 = np.einsum("abcd, ab -> cd", pdfter.four_overlap, vp_basis_all96)
# print("vp_all96 consists with vp_fock_all96?", np.allclose(vp_fock_all96_1, vp_fock_all96, atol=np.linalg.norm(vp_fock_all96)*0.1))
# vp_all96_1 = mol.to_grid(vp_basis_all96)
# print("basis and grid back and forth consistence?", np.allclose(vp_all96_1, vp_all96, atol=np.linalg.norm(vp_all96)*0.1))
# if not np.allclose(vp_all96_1, vp_all96, atol=np.linalg.norm(vp_all96)*0.1):
#     print("size of basis", mol.nbf)
#     print("The difference is", np.linalg.norm(vp_all96_1 - vp_all96)/np.linalg.norm(vp_all96))
#     pdft.plot1d_x(vp_all96_1, mol.Vpot, title="vp presented by the basis", fignum=5)

#%% Plot vp_all96 on grid.
# L = [5.0,  5.0, 4.0]
# D = [0.2, 0.2, 0.2]
# # Plot file
# O, N = libcubeprop.build_grid(mol.wfn, L, D)
# block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
# vp_basis_all96_psi4 = psi4.core.Matrix.from_array(vp_basis_all96)
# vp_cube = libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, vp_basis_all96_psi4)
# rotated_img = scipy.ndimage.rotate(vp_cube[:, :50, 19], -90)
# f, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=160)
# plt.imshow(rotated_img, interpolation="bicubic")
# plt.title("vpALL96 on basis.")
# plt.colorbar(fraction=0.040, pad=0.04)
# plt.savefig("vp2D")
