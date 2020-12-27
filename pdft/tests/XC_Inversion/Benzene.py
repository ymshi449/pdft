import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np
import pickle


if __name__ == "__main__":
    psi4.set_num_threads(2)
    psi4.set_memory('3 GB')
spherical_points = 350
radial_points = 35

input_density_wfn_method = "CCSD"
reference = "RHF"

functional = 'svwn'
basis = "cc-pcvdz"
vxc_basis = None

ortho_basis = False
svd = "input_once"
opt_method="trust-krylov"
method = "WuYangScipy"
v0 = "FermiAmaldi"

title = method +"\n"+ \
        basis + "/" + str(vxc_basis) + str(ortho_basis) + "\n" + \
        input_density_wfn_method + "\n" +\
        reference + "\n" + \
        "grid"+str(radial_points)+"/"+str(spherical_points)+"\n"+\
        v0 + "\n"\
        + opt_method + "_" + str(svd)
print(title)

psi4.set_output_file("Ne.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
C 

H 1 1.082 

C 1 1.387 2 120.0 

H 3 1.082 1 120.0 2 0.0 

C 3 1.387 1 120.0 4 180.0 

H 5 1.082 3 120.0 4 0.0 

C 5 1.387 3 120.0 6 180.0 

H 7 1.082 5 120.0 6 0.0 

C 7 1.387 5 120.0 8 180.0 

H 9 1.082 7 120.0 8 0.0 

C 9 1.387 7 120.0 10 180.0 

H 11 1.082 9 120.0 10 0.0 
units bohr
symmetry c1
""")

Full_Molec.set_name("Ne")

# Exact
Ne = np.genfromtxt('/home/yuming/PDFT/pdft/pdft/data/Atom0/ne.new8/Data')
Ne_xyz = np.concatenate((-np.flip(Ne[:, 1]), Ne[:, 1]))
Ne_vxc = np.concatenate((np.flip(Ne[:, 3]), Ne[:, 3]))
Ne_n = np.concatenate((np.flip(Ne[:, 2]), Ne[:, 2]))
#Psi4 Options:
psi4.set_options({
    'DFT_SPHERICAL_POINTS': spherical_points,
    'DFT_RADIAL_POINTS': radial_points,
    "opdm": True,
    "tpdm": True,
    "maxiter": 1000,
    'REFERENCE': reference,
    "CUBIC_BlOCK_MAX_POINTS": 10000
})

print("Target Density Calculation Started.")
#  Get wfn for target density
# E_input, input_density_wfn = psi4.energy("CCSD"+"/"+basis, molecule=Full_Molec, return_wfn=True)
# _, input_density_wfn = psi4.gradient("CCSD"+"/"+basis, molecule=Full_Molec, return_wfn=True)
# _,input_density_wfn = psi4.properties("CISD/"+basis, molecule=Full_Molec,
#                                             return_wfn=True, properties=['DIPOLE'])
if input_density_wfn_method.upper() == "DETCI":
    E_input,input_density_wfn = psi4.energy("DETCI/"+basis, molecule=Full_Molec,
                                            return_wfn=True)
elif input_density_wfn_method.upper() == "SVWN":
    E_input,input_density_wfn = psi4.energy("SVWN/"+basis, molecule=Full_Molec,
                                            return_wfn=True)
elif input_density_wfn_method.upper() == "SCF":
    E_HF, input_density_wfn = psi4.energy("SCF"+"/"+basis, molecule=Full_Molec, return_wfn=True)
elif input_density_wfn_method.upper() == "CCSD":
    _,input_density_wfn = psi4.properties("CCSD/"+basis, molecule=Full_Molec, properties=['dipole'], return_wfn=True)


print("Target Density Calculation Finished.")

mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf(100, print_energies=True)
print("The inverter molecule is constructed.")
if vxc_basis is not None:
    vxc_basis = XC_Inversion.Molecule(Full_Molec, vxc_basis, functional, jk="JK not needed")
    # vxc_basis.scf(100, print_energies=True)
else:
    vxc_basis = mol
print("The potential basis set constructed.")
print("Number of Basis: ", mol.nbf, vxc_basis.nbf)

inverser = XC_Inversion.Inverser(mol, input_density_wfn,
                                 ortho_basis=ortho_basis,
                                 vxc_basis=vxc_basis,
                                 v0=v0,
                                 # eHOMO=-0.5792,
                                 # v0_wfn=v0_wfn
                                 )

# inverser.find_vxc_scipy_WuYang(opt_method="trust-krylov")
inverser.find_vxc_scipy_constrainedoptimization(opt_method="L-BFGS-B");

# # %% Plotter
# L = [3, 0, 0]
# D = [1, 0.02, 0.02]
# O = [0, -4, -4]
# N = [1, 401, 401]
# inverser.v_output_a = inverser.v_output[:vxc_basis.nbf]
# vout_cube_a, xyzw = libcubeprop.basis_to_cubic_grid(inverser.v_output_a,
#                                                     inverser.vp_basis.wfn, L, D, O, N)
# vout_cube_a = np.squeeze(vout_cube_a)
# xyzw[0].shape = xyzw[0].shape[0] * xyzw[0].shape[1] * xyzw[0].shape[0] * xyzw[0].shape[2]
# xyzw[1].shape = xyzw[0].shape
# xyzw[2].shape = xyzw[0].shape
# xyzw[3].shape = xyzw[0].shape
# grid = np.array([xyzw[0][:], xyzw[1][:], xyzw[2][:]])
# if v0 == "FermiAmaldi":
#     if (inverser.vH4v0 is None) or  (inverser.vH4v0.shape != vout_cube_a.shape):
#         grid = grid.T
#         inverser.get_esp4v0(grid=grid)
#         inverser.get_vH_vext(grid)
#         inverser.vH4v0.shape = vout_cube_a.shape
#         grid = grid.T
#     nocc = mol.ndocc
#     inverser.vxc_a_grid = vout_cube_a - 1 / nocc * inverser.vH4v0
# elif v0 == "Hartree":
#     inverser.vxc_a_grid = vout_cube_a
#
# fig, ax = plt.subplots(dpi=200)
# im = ax.imshow(inverser.vxc_a_grid, interpolation="bicubic")
# fig.colorbar(im)
# fig.show()
#
# vxc_temp = np.copy(inverser.vxc_a_grid)
# vout_temp = np.copy(inverser.v_output_a)