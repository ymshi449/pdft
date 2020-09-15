import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

if __name__ == "__main__":
    psi4.set_num_threads(3)
    psi4.set_memory('4 GB')

functional = 'svwn'
basis = "cc-pvdz"
vxc_basis = "cc-pcvqz"

ortho_basis = False
svd = "input_once"
opt_method="trust-krylov"
method = "WuYangScipy"
v0 = "FermiAmaldi"

title = method +"_"+ opt_method +"_"+v0+ "_" + basis+"_"+ \
        str(vxc_basis) + "_"\
        + str(ortho_basis) + "_" + str(svd)
print(title)

psi4.set_output_file("Ne.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
Ne
units bohr
symmetry c1
""")

Full_Molec.set_name("Ne")

# Exact
Ne = np.genfromtxt('/home/yuming/PDFT/pdft/pdft/data/Atom0/ne.new8/Data')
Ne_xyz = np.concatenate((-np.flip(Ne[:, 1]), Ne[:, 1]))
Ne_vxc = np.concatenate((np.flip(Ne[:, 3]), Ne[:, 3]))

#Psi4 Options:
psi4.set_options({
    'DFT_SPHERICAL_POINTS': 302,
    'DFT_RADIAL_POINTS': 77,
    'MAXITER': 1000,
    'BASIS': basis,
    'REFERENCE': 'RHF'
})
#  Get wfn for target density
E_input, input_density_wfn = psi4.energy("CCSD"+"/"+basis, molecule=Full_Molec, return_wfn=True)
print("Target Density Calculation Finished.")

#Psi4 Options:
psi4.set_options({
    'REFERENCE' : 'UHF'
})
mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf_inversion(100)
if vxc_basis is not None:
    vxc_basis = XC_Inversion.Molecule(Full_Molec, vxc_basis, functional, jk=None)
    vxc_basis.scf_inversion(100)
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

# if method == "WuYangScipy":
#     inverser.find_vxc_scipy_WuYang(opt_method=opt_method)
# elif method == "WuYangMN":
#     hess, jac = inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="StrongWolfe")
# elif method == "COScipy":
#     inverser.find_vxc_scipy_constrainedoptimization(opt_method="L-BFGS-B");
#
# if method == "WuYangScipy":
#     inverser.find_vxc_scipy_WuYang(opt_method=opt_method)
# elif method == "WuYangMN":
#     hess, jac = inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="StrongWolfe")
# elif method == "COScipy":
#     inverser.find_vxc_scipy_constrainedoptimization(opt_method="L-BFGS-B");
#
# L = [3, 0, 0]
# D = [0.1, 0.5, 0.2]
# O = [-2.1, 0, 0]
# N = [100, 1, 1]
# inverser.v_output_a = inverser.v_output[:vxc_basis.nbf]
# vout_cube_a, xyzw = libcubeprop.basis_to_cubic_grid(inverser.v_output_a, inverser.vp_basis.wfn, L, D, O, N)
# vout_cube_a.shape = 100
# xyzw[0].shape = 100
# xyzw[1].shape = 100
# xyzw[2].shape = 100
# xyzw[3].shape = 100
# mark_y = np.isclose(xyzw[1], 0)
# mark_z = np.isclose(xyzw[2], 0)
# grid = np.array([xyzw[0][mark_y&mark_z], xyzw[1][mark_y&mark_z], xyzw[2][mark_y&mark_z]])
# grid = grid.T
# inverser.get_esp4v0(grid=grid)
# inverser.get_vH_vext(grid)
# nocc = mol.ndocc
# if v0 == "FermiAmaldi":
#     inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y] -1 / nocc * inverser.vH4v0
# elif v0 == "Hartree":
#     inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y]
# grid = grid.T
#
# f,ax = plt.subplots(1,1,dpi=200)
# ax.plot(Ne_xyz, Ne_vxc, label="Exact")
# XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, xyz=grid, ax=ax, label="$\lambda=0$", ls="--")
#
# rgl_list, L_list, dT_list, P_list = inverser.my_L_curve_regularization4WuYang();
# inverser.find_vxc_scipy_WuYang(opt_method=opt_method)
# inverser.v_output_a = inverser.v_output[:vxc_basis.nbf]
# vout_cube_a, _ = libcubeprop.basis_to_cubic_grid(inverser.v_output_a, inverser.vp_basis.wfn, L, D, O, N)
# vout_cube_a.shape = 100
# nocc = mol.ndocc
# if v0 == "FermiAmaldi":
#     inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y] -1 / nocc * inverser.vH4v0
# elif v0 == "Hartree":
#     inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y]
# XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, xyz=grid, ax=ax, label="$\lambda=%.2e$"%inverser.regularization_constant, ls="--")
#
# ax.set_xlim(-2.1, 8.1)
# ax.legend()
# f.show()
