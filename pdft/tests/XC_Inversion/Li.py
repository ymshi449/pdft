import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np
import pickle


if __name__ == "__main__":
    psi4.set_num_threads(1)
    psi4.set_memory('3 GB')
spherical_points = 350
radial_points = 35

input_density_wfn_method = "CCSD"
reference = "UHF"

functional = 'svwn'
basis = "cc-pcvdz"
vxc_basis = None

ortho_basis = False
svd = "input_once"
method = "WuYangScipy"
opt_method="trust-krylov"
v0 = "FermiAmaldi"

title = method +"\n"+ \
        basis + "/" + str(vxc_basis) + str(ortho_basis) + "\n" + \
        input_density_wfn_method + "\n" +\
        reference + "\n" + \
        "grid"+str(radial_points)+"/"+str(spherical_points)+"\n"+\
        v0 + "\n"\
        + opt_method + "_" + str(svd)
print(title)

psi4.set_output_file("Li.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
Li
units bohr
symmetry c1
""")

Full_Molec.set_name("Li")

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
    'save_jk': True
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
mol.scf(100)
if vxc_basis is not None:
    vxc_basis = XC_Inversion.Molecule(Full_Molec, vxc_basis, functional, jk=None)
    vxc_basis.scf(100)
else:
    vxc_basis = mol

print("Number of Basis: ", mol.nbf, vxc_basis.nbf)

inverser = XC_Inversion.Inverser(mol, input_density_wfn,
                                 ortho_basis=ortho_basis,
                                 vxc_basis=vxc_basis,
                                 v0=v0,
                                 # eHOMO=-0.5792,
                                 # v0_wfn=v0_wfn
                                 )

# opt = {
#     "disp": False,
#     "gtol": 1e-7,
# }
#
#
# inverser.regularization_constant = 4.5399929762484875e-05
#
#
# if method == "WuYangScipy":
#     inverser.find_vxc_scipy_WuYang(opt_method=opt_method, opt=opt);
# elif method == "WuYangMN":
#     hess, jac = inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="StrongWolfe");
# elif method == "COScipy":
#     inverser.find_vxc_scipy_constrainedoptimization(opt_method="L-BFGS-B");
#
# L = None
# D = [0.1, 0.02, 0.02]
# O = [-10, 0, 0]
# N = [201, 1, 1]
# # Alpha
# inverser.v_output_a = inverser.v_output[:vxc_basis.nbf]
# vout_cube_a, xyzw = libcubeprop.basis_to_cubic_grid(inverser.v_output_a,
#                                                     inverser.vp_basis.wfn, L, D, O, N)
# inverser.v_output_b = inverser.v_output[vxc_basis.nbf:]
# vout_cube_b, _ = libcubeprop.basis_to_cubic_grid(inverser.v_output_b,
#                                                     inverser.vp_basis.wfn, L, D, O, N)
# vout_cube_a = np.squeeze(vout_cube_a)
# vout_cube_b = np.squeeze(vout_cube_b)
# xyzw[0].shape = xyzw[0].shape[0] * xyzw[0].shape[1] * xyzw[0].shape[2]
# xyzw[1].shape = xyzw[0].shape
# xyzw[2].shape = xyzw[0].shape
# xyzw[3].shape = xyzw[0].shape
# grid = np.array([xyzw[0][:], xyzw[1][:], xyzw[2][:]])
# if v0 == "FermiAmaldi":
#     if (inverser.vH4v0 is None) or (inverser.vH4v0.shape != vout_cube_a.shape):
#         grid = grid.T
#         inverser.get_esp4v0(grid=grid)
#         inverser.get_vH_vext(grid)
#         inverser.vH4v0.shape = vout_cube_a.shape
#         grid = grid.T
#     nocc = mol.ndocc
#     inverser.vxc_a_grid = vout_cube_a - 1 / nocc * inverser.vH4v0
#     inverser.vxc_b_grid = vout_cube_b - 1 / nocc * inverser.vH4v0
# elif v0 == "Hartree":
#     inverser.vxc_a_grid = vout_cube_a
#     inverser.vxc_b_grid = vout_cube_b
#
# vxc_a = np.copy(inverser.vxc_a_grid)
# vout_a = np.copy(inverser.v_output_a)
# vxc_b = np.copy(inverser.vxc_b_grid)
# vout_b = np.copy(inverser.v_output_b)
#
# fig,ax = plt.subplots(1,1,dpi=200)
# XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, xyz=grid, ax=ax, label="alpha", ls="-")
# XC_Inversion.pdft.plot1d_x(inverser.vxc_b_grid, xyz=grid, ax=ax, label="beta", ls="--")
#
# # ax.set_xlim(-4, 8)
# ax.legend()
# fig.show()