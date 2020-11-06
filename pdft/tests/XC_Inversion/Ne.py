import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

if __name__ == "__main__":
    psi4.set_num_threads(2)

functional = 'svwn'
basis = 'cc-pvqz'

vxc_basis = None

ortho_basis = False
svd = "segment_cycle_cutoff"
opt_method="trust-krylov"
method = "WuYangScipy"
v0 = "Hartree"

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

#Psi4 Options:
psi4.set_options({
    'DFT_SPHERICAL_POINTS': 302,
    'DFT_RADIAL_POINTS': 77,
    'REFERENCE' : 'UHF'
})
E, input_density_wfn = psi4.energy(functional+"/"+basis, molecule=Full_Molec, return_wfn=True)
#Psi4 Options:
psi4.set_options({
    'REFERENCE' : 'UHF'
})
mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf_inversion(100)
if vxc_basis is not None:
    vxc_basis = XC_Inversion.Molecule(Full_Molec, vxc_basis, functional, jk="No jk needed")
    vxc_basis.scf_inversion(1)
else:
    vxc_basis = mol

print("Number of Basis: ", mol.nbf, vxc_basis.nbf)

inverser = XC_Inversion.Inverser(mol, input_density_wfn,
                                 ortho_basis=ortho_basis,
                                 vxc_basis=vxc_basis,
                                 v0=v0
                                 )

# grad, grad_app = inverser.check_gradient_constrainedoptimization()
# hess, hess_app = inverser.check_hess_constrainedoptimization()

# if method == "WuYangScipy":
#     inverser.find_vxc_scipy_WuYang(opt_method=opt_method)
# elif method == "WuYangMN":
#     # rcondlist, dnlist, Llist = inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="LD")
#     inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="StrongWolfeD")
# elif method == "COScipy":
#     inverser.find_vxc_scipy_constrainedoptimization(opt_method=opt_method)
#
# # dDa = input_density_wfn.Da().np - mol.Da.np
# # dDb = input_density_wfn.Db().np - mol.Db.np
# # dn = mol.to_grid(dDa + dDb)
#
# f,ax = plt.subplots(1,1,dpi=200)
# XC_Inversion.pdft.plot1d_x(inverser.input_vxc_a, input_density_wfn.V_potential(), ax=ax, label="LDA")
# XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, vxc_basis.Vpot, ax=ax, label="WuYang", ls='--')
# ax.legend()
# ax.set_xlim(1e-3, 10)
# ax.set_xscale("log")
# f.show()

#%%
rcond = 33  # QQ
# rcond = 24  # TT
GL_rcond = [rcond, -2, -2]
inverser.find_vxc_manualNewton(svd_rcond=rcond, line_search_method="StrongWolfeD", find_vxc_grid=False)
L = [3, 0, 0]
D = [0.1, 0.5, 0.2]
O = [-2.1, 0, 0]
N = [100, 1, 1]
inverser.v_output_a = inverser.v_output[:vxc_basis.nbf]
vout_cube_a, xyzw = libcubeprop.basis_to_cubic_grid(inverser.v_output_a,
                                                    inverser.vp_basis.wfn, L, D, O, N)
vout_cube_a.shape = 100
xyzw[0].shape = 100
xyzw[1].shape = 100
xyzw[2].shape = 100
xyzw[3].shape = 100
mark_y = np.isclose(xyzw[1], 0)
mark_z = np.isclose(xyzw[2], 0)
grid = np.array([xyzw[0][mark_y&mark_z], xyzw[1][mark_y&mark_z], xyzw[2][mark_y&mark_z]])
grid = grid.T
inverser.get_esp4v0(grid=grid)
inverser.get_vH_vext(grid)
nocc = mol.ndocc
if v0 == "FermiAmaldi":
    inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y] -1 / nocc * inverser.vH4v0
elif v0 == "Hartree":
    inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y]
grid = grid.T

f,ax = plt.subplots(1,1,dpi=200)
XC_Inversion.pdft.plot1d_x(inverser.input_vxc_a, input_density_wfn.V_potential(), ax=ax, label="LDA")
XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, xyz=grid, ax=ax, label="TSVD", ls="--")
ax.set_xlim(-2.1, 8.1)

vxc_TSVD = np.copy(inverser.vxc_a_grid)

inverser.find_vxc_manualNewton(svd_rcond="GL", line_search_method="StrongWolfeD", find_vxc_grid=False, svd_parameter=GL_rcond)

inverser.v_output_a = inverser.v_output[:vxc_basis.nbf]
v0_a = inverser.v0_output[:vxc_basis.nbf]
vbar_a = inverser.vbara_output[:vxc_basis.nbf]
vbar_b = inverser.vbarb_output[:vxc_basis.nbf]

vout_cube_a, _ = libcubeprop.basis_to_cubic_grid(inverser.v_output_a, inverser.vp_basis.wfn, L, D, O, N)
v0_cube_a, _ = libcubeprop.basis_to_cubic_grid(v0_a, inverser.vp_basis.wfn, L, D, O, N)
vbar_cube_a, _ = libcubeprop.basis_to_cubic_grid(vbar_a, inverser.vp_basis.wfn, L, D, O, N)
vbar_cube_b, _ = libcubeprop.basis_to_cubic_grid(vbar_b, inverser.vp_basis.wfn, L, D, O, N)
vout_cube_a.shape = 100
v0_cube_a.shape = 100
vbar_cube_a.shape = 100
vbar_cube_b.shape = 100

nocc = mol.ndocc
if v0 == "FermiAmaldi":
    inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y] -1 / nocc * inverser.vH4v0
    v0_a_grid = v0_cube_a[mark_z&mark_y] -1 / nocc * inverser.vH4v0
    vbar_a_grid = vbar_cube_a[mark_z&mark_y]
    vbar_b_grid = vbar_cube_b[mark_z&mark_y]
elif v0 == "Hartree":
    inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y]
    v0_a_grid = v0_cube_a[mark_z&mark_y]
    vbar_a_grid = vbar_cube_a[mark_z&mark_y]
    vbar_b_grid = vbar_cube_b[mark_z & mark_y]
XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, xyz=grid, ax=ax, label="TSVD+GL", ls="--")
XC_Inversion.pdft.plot1d_x(v0_a_grid, xyz=grid, ax=ax, label="v0", ls=":")
XC_Inversion.pdft.plot1d_x(vbar_a_grid, xyz=grid, ax=ax, label="vbara", ls=":")
XC_Inversion.pdft.plot1d_x(vbar_b_grid, xyz=grid, ax=ax, label="vbarb", ls=":")
ax.set_xlim(-2.1, 6)
ax.legend()
f.show()