import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

if __name__ == "__main__":
    psi4.set_num_threads(2)

spherical_points = 350
radial_points = 35

input_density_wfn_method = "SCF"
reference = "RHF"

functional = 'svwn'
basis = "cc-pcvqz"
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

psi4.set_output_file("FC2Cl.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
F -4.622780328 0 0
C -2.225662738 0 0
C 0        0 0
Cl 3.130803758 0 0
units bohr
symmetry c1
""")

Full_Molec.set_name("FC2Cl")

#Psi4 Options:
psi4.set_options({
    'DFT_SPHERICAL_POINTS': spherical_points,
    'DFT_RADIAL_POINTS': radial_points,
    "opdm": True,
    "tpdm": True,
    "maxiter": 1000,
    'REFERENCE': reference,
    "SAVE_JK": True
})

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

mol = XC_Inversion.Molecule(Full_Molec, basis, functional, jk=input_density_wfn.jk())
mol.scf(100)
if vxc_basis is not None:
    vxc_basis = XC_Inversion.Molecule(Full_Molec, vxc_basis, functional, jk="No Need for JK")
    print("Number of Basis: ", mol.nbf, vxc_basis.nbf)
    # assert vxc_basis.nbf < 230
    vxc_basis.scf(10)
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

# f,ax = plt.subplots(1,1,dpi=200)
# XC_Inversion.pdft.plot1d_x(inverser.input_vxc_a, input_density_wfn.V_potential(), ax=ax, label="LDA")
# XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, vxc_basis.Vpot, ax=ax, label="WuYang", ls='--')
# ax.legend()
# ax.set_xlim(-7.5,5.5)
# f.show()