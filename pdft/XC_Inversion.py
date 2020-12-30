"""
An inverser for vxc. Inherit from PDFT.
"""

import psi4
import numpy as np
import matplotlib.pyplot as plt
import pdft
import scipy.optimize as optimizer
import scipy.stats as stats
from lbfgs import fmin_lbfgs
# from scipy.linalg import eig as scipy_eig
import time
from opt_einsum import contract
if __name__ == "__main__":
    psi4.set_num_threads(2)


class Molecule(pdft.U_Molecule):
    def __init__(self, geometry, basis, method, omega=1, mints=None, jk=None):
        super().__init__(geometry, basis, method, omega=omega, mints=mints, jk=jk)

    def KS_solver(self, maxiter, V=None, print_energies=False, vp_matrix=None, add_vext=True):
        """
        Performs scf calculation to find energy and density with a given V
        Parameters
        ----------
        V_a: fock matrix of v_output_a + v_FA_a
        V_b: fock matrix of v_output_b + v_FA_b
        Returns
        -------
        """
        assert vp_matrix is None

        if V is None:
            V_a = psi4.core.Matrix.from_array(np.zeros_like(self.T.np))
            V_b = psi4.core.Matrix.from_array(np.zeros_like(self.T.np))
        else:
            V_a = V[0]
            V_b = V[1]

        if self.Da is None:
            C_a, Cocc_a, D_a, eigs_a = pdft.build_orbitals(self.H, self.A, self.nalpha)
            C_b, Cocc_b, D_b, eigs_b = pdft.build_orbitals(self.H, self.A, self.nbeta)
        else: # Use the calculation from last v as initial.
            nbf = self.A.shape[0]
            Cocc_a = psi4.core.Matrix(nbf, self.nalpha)
            Cocc_a.np[:] = self.Ca.np[:, :self.nalpha]
            Cocc_b = psi4.core.Matrix(nbf, self.nbeta)
            Cocc_b.np[:] = self.Cb.np[:, :self.nbeta]
            D_a = self.Da
            D_b = self.Db

        # diisa_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest")
        # diisb_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest")

        Eold = 0.0
        E = 0.0
        E_conv = psi4.core.get_option("SCF", "E_CONVERGENCE")
        E_conv = 1e-9
        D_conv = psi4.core.get_option("SCF", "D_CONVERGENCE")
        dRMS = -1
        for SCF_ITER in range(maxiter+1):
            #Bring core matrix
            if add_vext:
                F_a = self.H.clone()
                F_b = self.H.clone()
            else:
                F_a = self.T.clone()
                F_b = self.T.clone()
            F_a.axpy(1.0, V_a)
            F_b.axpy(1.0, V_b)

            # if V is not None:
            #     #DIIS
            #     diisa_e = psi4.core.triplet(F_a, D_a, self.S, False, False, False)
            #     diisa_e.subtract(psi4.core.triplet(self.S, D_a, F_a, False, False, False))
            #     diisa_e = psi4.core.triplet(self.A, diisa_e, self.A, False, False, False)
            #     diisa_obj.add(F_a, diisa_e)
            #
            #     diisb_e = psi4.core.triplet(F_b, D_b, self.S, False, False, False)
            #     diisb_e.subtract(psi4.core.triplet(self.S, D_b, F_b, False, False, False))
            #     diisb_e = psi4.core.triplet(self.A, diisb_e, self.A, False, False, False)
            #     diisb_obj.add(F_b, diisb_e)

            Core = 1.0 * self.H.vector_dot(D_a) + 1.0 * self.H.vector_dot(D_b)
            # This is not the correct Eks but Eks - Eext
            fake_Eks = V_a.vector_dot(D_a) + V_b.vector_dot(D_b)

            SCF_E = Core
            SCF_E += fake_Eks
            SCF_E += self.Enuc

            # if V is not None:
            #     dRMS = 0.5 * (np.mean(diisa_e.np**2)**0.5 + np.mean(diisb_e.np**2)**0.5)

            if V is not None:
                break
            elif (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
                if print_energies is True:
                    print(F'SCF Convergence: NUM_ITER = {SCF_ITER} dE = {abs(SCF_E - Eold)} dDIIS = {dRMS}')
                break

            Eold = SCF_E

            # if V is not None:
            #     #DIIS extrapolate
            #     F_a = diisa_obj.extrapolate()
            #     F_b = diisb_obj.extrapolate()

            #Diagonalize Fock matrix
            C_a, Cocc_a, D_a, eigs_a = pdft.build_orbitals(F_a, self.A, self.nalpha)
            C_b, Cocc_b, D_b, eigs_b = pdft.build_orbitals(F_b, self.A, self.nbeta)
            #Exchange correlation energy/matrix
            self.Vpot.set_D([D_a,D_b])
            self.Vpot.properties()[0].set_pointers(D_a, D_b)

            if SCF_ITER == maxiter:
                # raise Exception("Maximum number of SCF cycles exceeded.")
                print("Maximum number of SCF cycles exceeded.")
                if print_energies is True:
                    print(F'SCF Convergence: NUM_ITER = {SCF_ITER} dE = {abs(SCF_E - Eold)} dDIIS = {dRMS}')

        # Diagonalize Fock matrix
        C_a, Cocc_a, D_a, eigs_a = pdft.build_orbitals(F_a, self.A, self.nalpha)
        C_b, Cocc_b, D_b, eigs_b = pdft.build_orbitals(F_b, self.A, self.nbeta)
        # Exchange correlation energy/matrix
        self.Vpot.set_D([D_a, D_b])
        self.Vpot.properties()[0].set_pointers(D_a, D_b)

        Vks_a = psi4.core.Matrix.from_array(V_a.np)
        Vks_b = psi4.core.Matrix.from_array(V_b.np)
        Vks_a.axpy(1.0, self.V)
        Vks_b.axpy(1.0, self.V)
        self.Da             = D_a
        self.Db             = D_b
        self.energy         = SCF_E
        self.eig_a          = eigs_a
        self.eig_b          = eigs_b
        self.Fa             = F_a
        self.Fb             = F_b
        self.vks_a          = Vks_a
        self.vks_b          = Vks_b
        self.Ca             = C_a
        self.Cb             = C_b
        self.Cocca          = Cocc_a
        self.Coccb          = Cocc_b
        return

class Inverser(pdft.U_Embedding):
    def __init__(self, molecule, input_density_wfn, input_E=None, v0_wfn=None,
                 vxc_basis=None, eHOMO=None, ortho_basis=False,
                 v0="FermiAmaldi"):
        super().__init__([], molecule, vp_basis=vxc_basis)

        self.input_density_wfn = input_density_wfn
        self.input_E = input_E
        if v0_wfn is None:
            self.v0_wfn = self.input_density_wfn
        else:
            self.v0_wfn = v0_wfn

        # if eHOMO is None:
        #     if self.molecule.nalpha < self.molecule.nbeta:
        #         self.eHOMO = self.input_density_wfn.epsilon_b().np[self.molecule.nbeta-1]
        #     else:
        #         self.eHOMO = self.input_density_wfn.epsilon_a().np[self.molecule.nalpha-1]
        # else:
        self.eHOMO = eHOMO

        # v_output = [v_output_a, v_output_b]
        self.v_output = np.zeros(int(self.vp_basis.nbf)*2)
        self.v0_output = np.zeros(int(self.vp_basis.nbf)*2)
        self.vbara_output = np.zeros(int(self.vp_basis.nbf)*2)
        self.vbarb_output = np.zeros(int(self.vp_basis.nbf)*2)
        self.v_output_a = None
        self.v_output_b = None
        # From WuYang (25) vxc = v_output + vH[n_input-n] - 1/N*vH[n_input]
        self.vxc_a_grid = None
        self.vxc_b_grid = None

        # Vxc calculated on the grid
        self.v_output_grid = np.zeros(int(self.molecule.w.shape[0])*2)
        self.v_output_grid1 = np.zeros(self.molecule.nbf**2*2)

        # v_esp4v0 = - esp of input
        self.esp4v0 = None
        # self.get_esp4v0()

        # v_ext
        self.vext = None
        self.approximate_vext_cutoff = None  # Used as initial guess for vext calculation to cut the singularity.
        self.vH4v0 = None
        self.v0_Fock = None
        # self.get_vH_vext()
        self.vext_app = None  # vext calculated from WuYang'e method to replace the real vext to avoid singularities.

        # v0
        self.v0 = v0
        if self.v0 == "FermiAmaldi":
            self.get_FermiAmaldi_v0()
        elif self.v0 == "Hartree":
            self.get_Hartree_v0()

        # Get reference vxc
        self.vout_constant = 0.0  # A constant needed to compensate the constant potential shift

        self.input_vxc_a = None
        self.input_vxc_b = None
        self.input_vxc_cube = None # To be implemented.
        try:
            self.get_input_vxc()
        except:
            print("no input xc")
            # self.get_HartreeLDA_v0()

        # vH_mol
        self.vH_mol = None

        # Orthogonal basis set
        self.ortho_basis = ortho_basis

        # TSVD. The index of svd array from where to be cut.
        self.svd_index = None

        # Counter
        self.L_counter = 0
        self.grad_counter = 0
        self.hess_counter = 0

        # Fouroverlap of two different basis set (phi_i, phi_j, xi_u, xi_v)
        self.four_overlap_two_basis = None

        self.vxc_hole_WF = None

    def get_input_vxc(self):

        self.input_vxc_a, self.input_vxc_b = pdft.U_xc(self.input_density_wfn.Da().np, self.input_density_wfn.Db().np,
                                                       # self.molecule.Vpot)[-1]
                                                       self.input_density_wfn.V_potential())[-1]

    def get_esp4v0(self, grid=None, Vpot=None):
        if grid is None:
            if Vpot is None:
                grid = np.array(self.vp_basis.wfn.V_potential().get_np_xyzw()[:-1]).T
                grid = psi4.core.Matrix.from_array(grid)
                assert grid.shape[1] == 3
            else:
                grid = np.array(Vpot.get_np_xyzw()[:-1]).T
                grid = psi4.core.Matrix.from_array(grid)
                assert grid.shape[1] == 3

        nthreads = psi4.get_num_threads()
        psi4.set_num_threads(1)
        print("ESP fitting starts. This might take a while.")
        assert grid.shape[1] == 3, "Grid should be N*3 np.array"

        esp_calculator = psi4.core.ESPPropCalc(self.v0_wfn)

        psi4_grid = psi4.core.Matrix.from_array(grid)
        self.esp4v0 = - esp_calculator.compute_esp_over_grid_in_memory(psi4_grid).np
        print("ESP fitting done")
        psi4.set_num_threads(nthreads)
        return

    def get_mol_vH(self, grid=None, Vpot=None):
        self.molecule.update_wfn_info()
        nthreads = psi4.get_num_threads()
        psi4.set_num_threads(1)
        print("ESP fitting starts. This might take a while.")

        if grid is None:
            if Vpot is None:
                grid = np.array(self.vp_basis.wfn.V_potential().get_np_xyzw()[:-1]).T
                grid = psi4.core.Matrix.from_array(grid)
                assert grid.shape[1] == 3
            else:
                grid = np.array(Vpot.get_np_xyzw()[:-1]).T
                grid = psi4.core.Matrix.from_array(grid)
                assert grid.shape[1] == 3

        assert grid.shape[1] == 3, "Grid should be N*3 np.array"

        esp_calculator = psi4.core.ESPPropCalc(self.molecule.wfn)
        self.vH_mol = - esp_calculator.compute_esp_over_grid_in_memory(grid).np - self.vext

        print("ESP fitting done")
        psi4.set_num_threads(nthreads)
        return

    def get_vH_vext(self, grid=None, Vpot=None):

        # Get vext ---------------------------------------------------
        natom = self.molecule.wfn.molecule().natom()
        nuclear_xyz = self.molecule.wfn.molecule().full_geometry().np

        Z = np.zeros(natom)
        # index list of real atoms. To filter out ghosts.
        zidx = []
        for i in range(natom):
            Z[i] = self.molecule.wfn.molecule().Z(i)
            if Z[i] != 0:
                zidx.append(i)

        if grid is None:
            if Vpot is None:
                grid = np.array(self.vp_basis.wfn.V_potential().get_np_xyzw()[:-1]).T
                grid = psi4.core.Matrix.from_array(grid)
                assert grid.shape[1] == 3
            else:
                grid = np.array(Vpot.get_np_xyzw()[:-1]).T
                grid = psi4.core.Matrix.from_array(grid)
                assert grid.shape[1] == 3
        vext = np.zeros(grid.shape[0])
        # Go through all real atoms
        for i in range(len(zidx)):
            R = np.sqrt(np.sum((grid - nuclear_xyz[zidx[i], :])**2, axis=1))
            vext += Z[zidx[i]]/R
            vext[R < 1e-15] = 0

        self.vext = -vext

        # Get vH4v0 and vFermiAmaldi_fock ---------------------------------------------------
        self.vH4v0 = self.esp4v0 - self.vext
        # self.v0_Fock = self.molecule.grid_to_fock((nocc-1)/nocc*self.vH4v0)
        return

    def update_vout_constant(self):
        """
        This function works to shift the vout to make sure that eHOMO calculated is equal to eHOMO given
        :return:
        """
        if self.eHOMO is not None:
            if self.molecule.nalpha < self.molecule.nbeta:
                self.vout_constant = self.eHOMO - self.molecule.eig_b.np[self.molecule.nbeta-1]
            else:
                self.vout_constant = self.eHOMO - self.molecule.eig_a.np[self.molecule.nalpha-1]
            # print("Potential constant ", self.vout_constant)
        else:
            self.vout_constant = 0
        return

    def change_v0(self, v0: str):
        self.v0 = v0
        if self.v0 == "FermiAmaldi":
            self.get_FermiAmaldi_v0()
        elif self.v0 == "Hartree":
            self.get_Hartree_v0()
        else:
            assert False

    def change_orthogonality(self, ortho: bool):
        self.ortho_basis = ortho
        self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                         self.molecule.wfn.basisset(),
                                                                         self.vp_basis.wfn.basisset()))
        if self.ortho_basis:
            self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)


    def get_FermiAmaldi_v0(self):
        """
        Use v_FermiAmaldi as v0.
        :return:
        """
        self.v0 = "FermiAmaldi"
        # Grid to fock method
        nocc = self.molecule.ndocc
        # self.v0_Fock = (nocc-1)/nocc*self.molecule.grid_to_fock(self.vH4v0)

        Cocca = psi4.core.Matrix.from_array(self.v0_wfn.Ca().np[:,:self.molecule.nalpha])
        Coccb = psi4.core.Matrix.from_array(self.v0_wfn.Cb().np[:,:self.molecule.nbeta])
        self.molecule.jk.C_left_add(Cocca)
        self.molecule.jk.C_left_add(Coccb)
        self.molecule.jk.compute()
        self.molecule.jk.C_clear()

        self.v0_Fock = (nocc-1)/nocc*(self.molecule.jk.J()[0].np + self.molecule.jk.J()[1].np)
        return

    def get_Hartree_v0(self):
        """
        Use vH as v0.
        :return:
        """
        self.v0 = "Hartree"

        Cocca = psi4.core.Matrix.from_array(self.v0_wfn.Ca().np[:,:self.molecule.nalpha])
        Coccb = psi4.core.Matrix.from_array(self.v0_wfn.Cb().np[:,:self.molecule.nbeta])
        self.molecule.jk.C_left_add(Cocca)
        self.molecule.jk.C_left_add(Coccb)
        self.molecule.jk.compute()
        self.molecule.jk.C_clear()

        self.v0_Fock = (self.molecule.jk.J()[0].np + self.molecule.jk.J()[1].np)
        return

    def get_HartreeLDA_v0(self):
        """
        Use vH + vLDA as v0.
        :return:
        """
        # Da = np.copy(self.molecule.Da.np)
        # Db = np.copy(self.molecule.Db.np)
        # self.molecule.Da.np[:] = np.copy(self.input_density_wfn.Da().np)
        # self.molecule.Db.np[:] = np.copy(self.input_density_wfn.Db().np)
        self.v0 = "HartreeLDA"
        print("Get Hartree+LDA as v0")
        # Get Hartree
        Cocca = psi4.core.Matrix.from_array(self.v0_wfn.Ca().np[:,:self.molecule.nalpha])
        Coccb = psi4.core.Matrix.from_array(self.v0_wfn.Cb().np[:,:self.molecule.nbeta])
        self.molecule.jk.C_left_add(Cocca)
        self.molecule.jk.C_left_add(Coccb)
        self.molecule.jk.compute()
        self.molecule.jk.C_clear()

        self.v0_Fock = (self.molecule.jk.J()[0].np + self.molecule.jk.J()[1].np)
        # Get LDA
        if self.input_vxc_a is None:
            self.molecule.Vpot.set_D([self.input_density_wfn.Da(), self.input_density_wfn.Db()])
            self.molecule.Vpot.properties()[0].set_pointers(self.input_density_wfn.Da(), self.input_density_wfn.Db())
            self.input_vxc_a, self.input_vxc_b = pdft.U_xc(self.input_density_wfn.Da().np, self.input_density_wfn.Db().np,
                                                           self.molecule.Vpot)[-1]
            self.molecule.Vpot.set_D([self.molecule.Da, self.molecule.Db])
            self.molecule.Vpot.properties()[0].set_pointers(self.molecule.Da, self.molecule.Db)

            self.v0_Fock += self.molecule.grid_to_fock(self.input_vxc_a)
        else:
            self.v0_Fock += self.molecule.grid_to_fock(self.input_vxc_a)

        assert self.molecule.nbeta == self.molecule.nalpha, "Currently can only handle close shell."
        assert np.allclose(self.input_vxc_a, self.input_vxc_b), "Currently can only handle close shell."

        return

    def get_HartreeLDAappext_v0(self):
        """
        Use vH + vLDA + approximate_vext as v0.
        :return:
        """
        # Da = np.copy(self.molecule.Da.np)
        # Db = np.copy(self.molecule.Db.np)
        # self.molecule.Da.np[:] = np.copy(self.input_density_wfn.Da().np)
        # self.molecule.Db.np[:] = np.copy(self.input_density_wfn.Db().np)
        self.v0 = "HartreeLDAappext"
        print("Get Hartree+LDA+approximate_vext as v0")
        # Get Hartree
        Cocca = psi4.core.Matrix.from_array(self.v0_wfn.Ca().np[:,:self.molecule.nalpha])
        Coccb = psi4.core.Matrix.from_array(self.v0_wfn.Cb().np[:,:self.molecule.nbeta])
        self.molecule.jk.C_left_add(Cocca)
        self.molecule.jk.C_left_add(Coccb)
        self.molecule.jk.compute()
        self.molecule.jk.C_clear()

        self.v0_Fock = (self.molecule.jk.J()[0].np + self.molecule.jk.J()[1].np)
        # Get LDA
        if self.input_vxc_a is None:
            self.molecule.Vpot.set_D([self.input_density_wfn.Da(), self.input_density_wfn.Db()])
            self.molecule.Vpot.properties()[0].set_pointers(self.input_density_wfn.Da(), self.input_density_wfn.Db())
            self.input_vxc_a, self.input_vxc_b = pdft.U_xc(self.input_density_wfn.Da().np, self.input_density_wfn.Db().np,
                                                           self.molecule.Vpot)[-1]
            self.molecule.Vpot.set_D([self.molecule.Da, self.molecule.Db])
            self.molecule.Vpot.properties()[0].set_pointers(self.molecule.Da, self.molecule.Db)

            self.v0_Fock += self.molecule.grid_to_fock(self.input_vxc_a)
        else:
            self.v0_Fock += self.molecule.grid_to_fock(self.input_vxc_a)

        # # Get approximate vext
        if self.approximate_vext_cutoff is None:
            self.approximate_vext_cutoff = -10
        natom = self.molecule.wfn.molecule().natom()
        nuclear_xyz = self.molecule.wfn.molecule().full_geometry().np
        Z = np.zeros(natom)
        # index list of real atoms. To filter out ghosts.
        zidx = []
        for i in range(natom):
            Z[i] = self.molecule.wfn.molecule().Z(i)
            if Z[i] != 0:
                zidx.append(i)

        grid = np.array(self.molecule.Vpot.get_np_xyzw()[:-1]).T
        grid = psi4.core.Matrix.from_array(grid)
        vext = np.zeros(grid.shape[0])
        # Go through all real atoms
        for i in range(len(zidx)):
            R = np.sqrt(np.sum((grid - nuclear_xyz[zidx[i], :])**2, axis=1))
            vext += Z[zidx[i]]/R
            vext[R < 1e-15] = 0
        approximate_vext = -vext
        approximate_vext[approximate_vext<=self.approximate_vext_cutoff] = self.approximate_vext_cutoff
        self.v0_Fock += self.molecule.grid_to_fock(approximate_vext)

        # Get vext
        # self.v0_Fock += self.molecule.V.np

        assert self.molecule.nbeta == self.molecule.nalpha, "Currently can only handle close shell."
        assert np.allclose(self.input_vxc_a, self.input_vxc_b), "Currently can only handle close shell."
        return


    def get_vxc(self):
        """
        WuYang (25)
        """
        nbf = int(self.v_output.shape[0] / 2)
        self.v_output_a = self.v_output[:nbf]
        self.v_output_b = self.v_output[nbf:]

        # Get vH_mol
        self.get_mol_vH()

        nocc = self.molecule.ndocc
        if self.ortho_basis:
            self.vxc_a_grid = self.vp_basis.to_grid(np.dot(self.vp_basis.A.np, self.v_output_a))
            self.vxc_b_grid = self.vp_basis.to_grid(np.dot(self.vp_basis.A.np, self.v_output_b))
        else:
            self.vxc_a_grid = self.vp_basis.to_grid(self.v_output_a)
            self.vxc_b_grid = self.vp_basis.to_grid(self.v_output_b)

        if self.v0 == "FermiAmaldi":
            self.vxc_a_grid += (nocc-1)/nocc*self.vH4v0 - self.vH_mol
            self.vxc_b_grid += (nocc-1)/nocc*self.vH4v0 - self.vH_mol

        elif self.v0 == "Hartree":
            self.vxc_a_grid += self.vH4v0 - self.vH_mol
            self.vxc_b_grid += self.vH4v0 - self.vH_mol
        elif self.v0 == "HartreeLDA":
            # vH_in + vLDA_in + vout = vxc + vH  =>  vxc = vout + vH_in - vH + vLDA
            self.vxc_a_grid += self.vH4v0 - self.vH_mol + self.input_vxc_a
            self.vxc_b_grid += self.vH4v0 - self.vH_mol + self.input_vxc_b

        self.vxc_a_grid += self.vout_constant
        self.vxc_b_grid += self.vout_constant
        return

    def get_vxc_grid(self):
        """
        WuYang (25)
        """
        nbf = int(self.v_output_grid.shape[0] / 2)
        self.v_output_a = self.v_output_grid[:nbf]
        self.v_output_b = self.v_output_grid[nbf:]

        self.vxc_a_grid = np.copy(self.v_output_a)
        self.vxc_b_grid = np.copy(self.v_output_b)

        # Get vH_mol
        self.get_mol_vH()

        nocc = self.molecule.ndocc

        if self.v0 == "FermiAmaldi":
            self.vxc_a_grid += (nocc-1)/nocc*self.vH4v0 - self.vH_mol
            self.vxc_b_grid += (nocc-1)/nocc*self.vH4v0 - self.vH_mol

        elif self.v0 == "Hartree":
            self.vxc_a_grid += self.vH4v0 - self.vH_mol
            self.vxc_b_grid += self.vH4v0 - self.vH_mol
        elif self.v0 == "HartreeLDA":
            # vH_in + vLDA_in + vout = vxc + vH  =>  vxc = vout + vH_in - vH + vLDA
            self.vxc_a_grid += self.vH4v0 - self.vH_mol + self.input_vxc_a
            self.vxc_b_grid += self.vH4v0 - self.vH_mol + self.input_vxc_b

        self.vxc_a_grid += self.vout_constant
        self.vxc_b_grid += self.vout_constant
        return

    def Lagrangian_WuYang(self, v=None, no_tuple_return=True, fit4vxc=True):
        """
        L = - <T> - \int (vks_a*(n_a-n_a_input)+vks_b*(n_b-n_b_input))
        :return: L
        """
        self.L_counter += 1

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        if v is not None:
            nbf = int(v.shape[0]/2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            self.molecule.KS_solver(1000, [Vks_a, Vks_b], add_vext=fit4vxc)
            self.update_vout_constant()
            # update self.vout_constant

        L = - self.molecule.T.vector_dot(self.molecule.Da) - self.molecule.T.vector_dot(self.molecule.Db)
        L += - self.molecule.vks_a.vector_dot(self.molecule.Da) - self.molecule.vks_b.vector_dot(self.molecule.Db)
        L += self.molecule.vks_a.vector_dot(self.input_density_wfn.Da()) + self.molecule.vks_b.vector_dot(self.input_density_wfn.Db())

        if self.regularization_constant is not None:
            T = self.vp_basis.T.np
            if v is not None:
                norm = 2 * np.dot(np.dot(v_output_a, T), v_output_a) + 2 * np.dot(np.dot(v_output_b, T), v_output_b)
            else:
                nbf = int(self.v_output.shape[0] / 2)
                norm = 2 * np.dot(np.dot(self.v_output[:nbf], T), self.v_output[:nbf]) + \
                       2 * np.dot(np.dot(self.v_output[nbf:], T), self.v_output[nbf:])
            L += norm * self.regularization_constant
            self.regul_norm = norm

        return L

    def grad_WuYang(self, v=None, no_tuple_return=True, fit4vxc=True):
        """
        grad_a = dL/dvxc_a = - (n_a-n_a_input)
        grad_b = dL/dvxc_b = - (n_b-n_b_input)
        :return:
        """
        self.grad_counter += 1

        if v is not None:
            nbf = int(v.shape[0] / 2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a)
                                                + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b)
                                                + self.vout_constant * self.molecule.S.np + self.v0_Fock)

            self.molecule.KS_solver(100, [Vks_a, Vks_b], add_vext=fit4vxc)
            self.update_vout_constant()

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np

        grad_a = np.einsum("uv,uvi->i", dDa, self.three_overlap, optimize=True)
        grad_b = np.einsum("uv,uvi->i", dDb, self.three_overlap, optimize=True)
        grad = np.concatenate((grad_a, grad_b))

        # Regularization
        if self.regularization_constant is not None:
            T = self.vp_basis.T.np
            if v is not None:
                grad[:nbf] += 4*self.regularization_constant*np.dot(T, v_output_a)
                grad[nbf:] += 4*self.regularization_constant*np.dot(T, v_output_b)
            else:
                nbf = int(self.v_output.shape[0] / 2)
                grad[:nbf] += 4*self.regularization_constant*np.dot(T, self.v_output[:nbf])
                grad[nbf:] += 4*self.regularization_constant*np.dot(T, self.v_output[nbf:])

        return grad

    def hess_WuYang(self, v=None, fit4vxc=True):
        """
        hess: WuYang (21)
        """
        self.hess_counter += 1
        if v is not None:
            nbf = int(v.shape[0] / 2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)

            self.molecule.KS_solver(100, [Vks_a, Vks_b], add_vext=fit4vxc)
            self.update_vout_constant()

        epsilon_occ_a = self.molecule.eig_a.np[:self.molecule.nalpha, None]
        epsilon_occ_b = self.molecule.eig_b.np[:self.molecule.nbeta, None]
        epsilon_unocc_a = self.molecule.eig_a.np[self.molecule.nalpha:]
        epsilon_unocc_b = self.molecule.eig_b.np[self.molecule.nbeta:]
        epsilon_a = epsilon_occ_a - epsilon_unocc_a
        epsilon_b = epsilon_occ_b - epsilon_unocc_b

        hess = np.zeros((self.vp_basis.nbf*2, self.vp_basis.nbf*2))

        # Alpha electrons
        hess[0:self.vp_basis.nbf, 0:self.vp_basis.nbf] = - self.molecule.omega * np.einsum('ai,bj,ci,dj,ij,abm,cdn -> mn',
                                                                                             self.molecule.Ca.np[:, :self.molecule.nalpha],
                                                                                             self.molecule.Ca.np[:, self.molecule.nalpha:],
                                                                                             self.molecule.Ca.np[:, :self.molecule.nalpha],
                                                                                             self.molecule.Ca.np[:, self.molecule.nalpha:],
                                                                                             np.reciprocal(epsilon_a), self.three_overlap,
                                                                                             self.three_overlap, optimize=True)
        # Beta electrons
        hess[self.vp_basis.nbf:, self.vp_basis.nbf:] = - self.molecule.omega * np.einsum('ai,bj,ci,dj,ij,abm,cdn -> mn',
                                                                                           self.molecule.Cb.np[:, :self.molecule.nbeta],
                                                                                           self.molecule.Cb.np[:, self.molecule.nbeta:],
                                                                                           self.molecule.Cb.np[:, :self.molecule.nbeta],
                                                                                           self.molecule.Cb.np[:, self.molecule.nbeta:],
                                                                                           np.reciprocal(epsilon_b),self.three_overlap,
                                                                                           self.three_overlap, optimize=True)
        hess = (hess + hess.T)

        # Regularization
        if self.regularization_constant is not None:
            T = self.vp_basis.T.np
            T = 0.5 * (T + T.T)
            hess[self.vp_basis.nbf:, self.vp_basis.nbf:] += 4 * self.regularization_constant*T
            hess[0:self.vp_basis.nbf, 0:self.vp_basis.nbf] += 4 * self.regularization_constant*T

        return hess

    def yang_L_curve_regularization4WuYang(self, rgl_bs=np.e, rgl_epn=15, scipy_opt_method="trust-krylov", print_flag=False):
        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        v_zero_initial = np.zeros_like(self.v_output)

        L_list = []
        rgl_order = np.array(range(rgl_epn))
        rgl_list = rgl_bs**-(rgl_order+2)
        rgl_list = np.append(rgl_list, 0)
        norm_list = []
        E_list = []
        T = self.vp_basis.T.np

        print("Start L-curve search for regularization constant lambda. This might take a while..")
        for regularization_constant in rgl_list:
            print(regularization_constant)
            self.regularization_constant = regularization_constant
            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.KS_solver(100, [Vks_a, Vks_b])
            self.update_vout_constant()

            opt = {
                "disp": False,
                "maxiter": 10000,
                # "eps": 1e-7
                # "norm": 2,
                # "gtol": 1e-7
            }

            v_result = optimizer.minimize(self.Lagrangian_WuYang, v_zero_initial,
                                          jac=self.grad_WuYang,
                                          hess=self.hess_WuYang,
                                          method=scipy_opt_method,
                                          options=opt)

            v = v_result.x
            nbf = int(v.shape[0]/2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]
            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            self.molecule.KS_solver(1000, [Vks_a, Vks_b])
            self.update_vout_constant()

            E_list.append(self.molecule.energy)


            norm = 2 * np.dot(np.dot(v_output_a, T), v_output_a) + \
                   2 * np.dot(np.dot(v_output_b, T), v_output_b)
            L_list.append(v_result.fun - norm * self.regularization_constant)
            norm_list.append(norm)
            if print_flag:
                print("=============L-curve, lambda: %e, W %f, reg %f==============="
                      %(self.regularization_constant, L_list[-1], norm_list[-1]))

        x = np.abs(L_list[:-1] - L_list[-1])
        y = norm_list[:-1]
        drv = x / (y * rgl_list[:-1])

        self.regularization_constant = rgl_list[np.argmin(drv)]
        print("Regularization constant lambda from L-curve is ", self.regularization_constant)

        if print_flag:


            f, ax = plt.subplots(1, 1, dpi=200)
            ax.scatter(np.log10(x), np.log10(y), s=1)
            ax.scatter(np.log10(x), 1./drv, s=1)
            ax.plot(np.log10(x), 1./drv, linewidth=1)
            # idx=0
            # for i, j in zip(x, y):
            #     ax.annotate("%.1e"%rgl_list[idx], xy=(i, j*0.9))
            #     # ax.annotate("%.1e"%drv[idx-1], xy=(i, j*1.1))
            #     idx += 1
            f.show()
        return rgl_list, L_list, norm_list, E_list

    def my_L_curve_regularization4WuYang(self, rgl_bs=np.e, rgl_epn=15, starting_epn=1,
                                  searching_method="close_to_platform",
                                  close_to_platform_rtol=0.001,
                                  scipy_opt_method="trust-krylov", print_flag=True):

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)
        v_zero_initial = np.zeros_like(self.v_output)

        # Loop
        L_list = []
        dT_list = []
        P_list = []
        rgl_order = starting_epn + np.array(range(rgl_epn))
        rgl_list =  rgl_bs**-(rgl_order+2)
        rgl_list = np.append(rgl_list, 0)
        n_input = self.molecule.to_grid(self.input_density_wfn.Da().np + self.input_density_wfn.Db().np)

        print("Start L-curve search for regularization constant lambda. This might take a while..")
        for regularization_constant in rgl_list:
            self.regularization_constant = regularization_constant
            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.KS_solver(100, [Vks_a, Vks_b])
            self.update_vout_constant()

            opt = {
                "disp": False,
                "maxiter": 10000,
            }

            v_result = optimizer.minimize(self.Lagrangian_WuYang, v_zero_initial,
                                          jac=self.grad_WuYang,
                                          hess=self.hess_WuYang,
                                          method=scipy_opt_method,
                                          options=opt)

            v = v_result.x
            # _ = self.find_vxc_manualNewton(svd_rcond=113*2, line_search_method="StrongWolfe")
            # v = self.v_output
            nbf = int(v.shape[0]/2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]
            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            self.molecule.KS_solver(1000, [Vks_a, Vks_b])
            self.update_vout_constant()

            n_result = self.molecule.to_grid(self.molecule.Da.np + self.molecule.Db.np)
            P = np.linalg.norm(v_output_a) * np.sum(np.abs(n_result - n_input) * self.molecule.w)
            P_list.append(P)
            # L_list.append(v_result.fun - self.regul_norm * self.regularization_constant)
            dT_list.append(self.molecule.T.vector_dot(self.molecule.Da)
                           + self.molecule.T.vector_dot(self.molecule.Db))
            print(regularization_constant, "P", P, "T", dT_list[-1])

        # L-curve
        dT_list = np.array(dT_list)

        f, ax = plt.subplots(1, 1, dpi=200)
        ax.scatter(range(dT_list.shape[0]), dT_list)
        f.show()
        if searching_method == "std_error_min":
            start_idx = int(input("Enter start index for the left line of L-curve: "))

            r_list = []
            std_list = []
            for i in range(start_idx + 3, dT_list.shape[0] - 2):
                left = dT_list[start_idx:i]
                right = dT_list[i + 1:]
                left_x = range(start_idx, i)
                right_x = range(i + 1, dT_list.shape[0])
                slopl, intl, rl, _, stdl = stats.linregress(left_x, left)
                slopr, intr, rr, _, stdr = stats.linregress(right_x, right)
                if print_flag:
                    print(i, stdl + stdr, rl + rr, "Right:", slopr, intr, "Left:", slopl, intl)
                r_list.append(rl + rr)
                std_list.append(stdl + stdr)

            # The final index
            i = np.argmin(std_list) + start_idx + 3
            self.regularization_constant = rgl_list[i]
            print("Regularization constant lambda from L-curve is ", self.regularization_constant)

            if print_flag:
                left = dT_list[start_idx:i]
                right = dT_list[i + 1:]
                left_x = range(start_idx, i)
                right_x = range(i + 1, dT_list.shape[0])
                slopl, intl, rl, _, stdl = stats.linregress(left_x, left)
                slopr, intr, rr, _, stdr = stats.linregress(right_x, right)
                x = np.array(ax.get_xlim())
                yl = intl + slopl * x
                yr = intr + slopr * x
                ax.plot(x, yl, '--')
                ax.plot(x, yr, '--')
                ax.set_ylim(np.min(dT_list)*0.99, np.max(dT_list)*1.01)
                f.show()

        elif searching_method == "close_to_platform":
            for i in range(len(dT_list)):
                if np.abs(dT_list[i] - dT_list[-1])/dT_list[-1] <= close_to_platform_rtol:
                    self.regularization_constant = rgl_list[i]
                    print("Regularization constant lambda from L-curve is ", self.regularization_constant)
                    break
            if print_flag:
                ax.axhline(y=dT_list[-1], ls="--", lw=0.7, color='r')
                ax.scatter(i, dT_list[i], marker="+")
                f.show()
        return rgl_list, L_list, dT_list, P_list

    def check_gradient_WuYang(self, dv=None):
        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        nbf = int(self.v_output.shape[0] / 2)
        v_output_a = self.v_output[:nbf]
        v_output_b = self.v_output[nbf:]

        Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a)
                                            + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b)
                                            + self.vout_constant * self.molecule.S.np + self.v0_Fock)

        self.molecule.KS_solver(100, [Vks_a, Vks_b])
        self.update_vout_constant()

        L = self.Lagrangian_WuYang()
        grad = self.grad_WuYang()

        if dv is None:
            dv = 1e-7*np.ones_like(self.v_output)

        grad_app = np.zeros_like(dv)

        for i in range(dv.shape[0]):
            dvi = np.zeros_like(dv)
            dvi[i] = dv[i]

            L_new = self.Lagrangian_WuYang(dvi+self.v_output)

            grad_app[i] = (L_new-L) / dvi[i]

        print(np.sum(grad*grad_app)/np.linalg.norm(grad)/np.linalg.norm(grad_app))
        print(np.linalg.norm(grad_app-grad)/np.linalg.norm(grad))

        return grad, grad_app

    def check_hess_WuYang(self, dv=None):
        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:

                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        nbf = int(self.v_output.shape[0] / 2)
        v_output_a = self.v_output[:nbf]
        v_output_b = self.v_output[nbf:]

        Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        self.molecule.KS_solver(100, [Vks_a, Vks_b])
        self.update_vout_constant()

        hess = self.hess_WuYang()
        grad = self.grad_WuYang()

        if dv is None:
            dv = 1e-7*np.ones_like(self.v_output)

        hess_app = np.zeros_like(hess)

        for i in range(grad.shape[0]):
            dvi = np.zeros_like(dv)
            dvi[i] = dv[i]

            grad_new = self.grad_WuYang(dvi+self.v_output)

            hess_app[i,:] = (grad_new - grad)/dv[i]

        hess_app = 0.5 * (hess_app + hess_app.T)
        print(np.trace(hess_app.dot(hess.T))/np.linalg.norm(hess_app)/np.linalg.norm(hess))
        print(np.linalg.norm(hess - hess_app)/np.linalg.norm(hess))

        return hess, hess_app

    def find_vxc_scipy_WuYang(self, maxiter=14000, opt_method="BFGS", opt=None, tol=None,
                              continue_opt=False, find_vxc_grid=True):

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:

                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)
        if not continue_opt:
            print("Zero the old result for a new calculation..")
            self.v_output = np.zeros_like(self.v_output)

            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.KS_solver(100, [Vks_a, Vks_b])
            self.update_vout_constant()

        print("<<<<<<<<<<<<<<<<<<<<<<WuYang vxc Inversion %s<<<<<<<<<<<<<<<<<<<"%opt_method)

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("|n| before", np.sum(np.abs(dn)*self.molecule.w))
        if opt is None:
            opt = {
                "disp": False,
                # "eps": 1e-7
                # "norm": 2,
                "gtol": 1e-2
            }
        opt["maxiter"] = maxiter

        result_x = optimizer.minimize(self.Lagrangian_WuYang, self.v_output,
                                      jac=self.grad_WuYang,
                                      hess=self.hess_WuYang,
                                      method=opt_method,
                                      options=opt,
                                      tol=tol)
        nbf = int(result_x.x.shape[0] / 2)
        v_output_a = result_x.x[:nbf]
        v_output_b = result_x.x[nbf:]

        Vks_a = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        self.molecule.KS_solver(100, [Vks_a, Vks_b])
        self.update_vout_constant()

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print(result_x.message)
        print("Evaluation: ", result_x.nfev)
        print("|jac|", np.linalg.norm(result_x.jac), "|n|", np.sum(np.abs(dn)*self.molecule.w), "L after", result_x.fun)
        print("Ts", self.molecule.Da.vector_dot(self.molecule.T)+self.molecule.Db.vector_dot(self.molecule.T))
        print("dTs", np.trace(np.dot(self.input_density_wfn.Da().np+self.input_density_wfn.Db().np-
                                     self.molecule.Da.np-self.molecule.Db.np, self.molecule.T.np)))
        print("eigenA")
        print(self.input_density_wfn.epsilon_a().np[:self.molecule.nalpha])
        print(self.molecule.eig_a.np[:self.molecule.nalpha])
        print("wfnDiff", self.input_density_wfn.Ca().vector_dot(self.molecule.Ca)
              /np.linalg.norm(self.input_density_wfn.Ca().np)/np.linalg.norm(self.molecule.Ca.np))
        print("Constant potential: ", self.vout_constant)

        # Update info
        self.v_output = result_x.x

        # if find_vxc_grid:
        #     self.get_vxc()
        return

    # def get_dvp_GL12modified(self, grad, hess, rcond=1e-6, rcond_bar=1e-6):
    #     """
    #     To get v0 and v_bar based on logic from Gidopoulos, Lathiotakis, 2012 PRA 85, 052508.
    #     BUT MODIFIED)
    #     :param grad:
    #     :param hess:
    #     :param rcond:
    #     :return:
    #     """
    #     nbf = int(grad.shape[0]/2)
    #     grad_up = grad[:nbf]
    #     grad_dn = grad[nbf:]
    #     hess_up = hess[:nbf,:nbf]
    #     hess_dn = hess[nbf:,nbf:]
    #
    #     Uup, sup, VTup = np.linalg.svd(hess_up)
    #     Udn, sdn, VTdn = np.linalg.svd(hess_dn)
    #
    #     filterup = sup < np.max(sup) * rcond
    #     filterdn = sdn < np.max(sdn) * rcond
    #
    #     b0up = Uup[:, np.bitwise_not(filterup)] @ VTup[np.bitwise_not(filterup), :] @ grad_up
    #     b0dn = Udn[:, np.bitwise_not(filterdn)] @ VTdn[np.bitwise_not(filterdn), :] @ grad_dn
    #     btdup = grad_up - b0up
    #     btddn = grad_dn - b0dn
    #
    #     sinv0up = 1/sup
    #     sinv0dn = 1/sdn
    #     sinv0up[filterup] = 0.0
    #     sinv0dn[filterdn] = 0.0
    #
    #     v0up = VTup.T @ np.diag(sinv0up) @ Uup.T @ b0up
    #     v0dn = VTdn.T @ np.diag(sinv0dn) @ Udn.T @ b0dn
    #
    #     v0 = np.concatenate((v0up, v0dn))
    #
    #     if self.four_overlap_two_basis is None:
    #         self.four_overlap_two_basis = pdft.fouroverlap([self.molecule.wfn, self.vp_basis.wfn],
    #                                              self.molecule.geometry, None, self.molecule.mints)[0]
    #
    #     A_full_up = np.einsum("ijuv,im,jm->uv", self.four_overlap_two_basis,
    #                           self.molecule.Cocca.np, self.molecule.Cocca.np, optimize=True)
    #     A_full_dn = np.einsum("ijuv,im,jm->uv", self.four_overlap_two_basis,
    #                           self.molecule.Coccb.np, self.molecule.Coccb.np, optimize=True)
    #
    #     Atdup = A_full_up - np.einsum('ai,bj,ci,dj,abm,cdn -> mn',
    #                                  self.molecule.Ca.np[:, :self.molecule.nalpha],
    #                                  self.molecule.Ca.np,
    #                                  self.molecule.Ca.np[:, :self.molecule.nalpha],
    #                                  self.molecule.Ca.np,
    #                                  self.three_overlap,
    #                                  self.three_overlap, optimize=True)
    #
    #     Atddn = A_full_dn - np.einsum('ai,bj,ci,dj,abm,cdn -> mn',
    #                                  self.molecule.Cb.np[:, :self.molecule.nbeta],
    #                                  self.molecule.Cb.np,
    #                                  self.molecule.Cb.np[:, :self.molecule.nbeta],
    #                                  self.molecule.Cb.np,
    #                                  self.three_overlap,
    #                                  self.three_overlap, optimize=True)
    #
    #     # # Rescaling
    #     sfullup = np.linalg.svd(A_full_up)[1]
    #     sfulldn = np.linalg.svd(A_full_dn)[1]
    #
    #     Atdup *= sup[0] / sfullup[0] / 2
    #     Atddn *= sdn[0] / sfulldn[0] / 2
    #
    #     # 1
    #     # vbarup_a = np.linalg.pinv(Atdup @ VTup[filterup, :].T, rcond=rcond) @ Uup[:, filterup] @ Uup[:, filterup].T \
    #     #            @ btdup @ VTup[filterup, :]
    #     # vbardn_a = np.linalg.pinv(Atddn @ VTdn[filterdn, :].T, rcond=rcond) @ Udn[:, filterdn] @ Udn[:, filterdn].T \
    #     #            @ btddn @ VTdn[filterdn, :]
    #     #
    #     # vbarup_b = - np.linalg.pinv(Atdup @ VTup[filterup, :].T, rcond=rcond) @ Uup[:, filterup] @ Uup[:, filterup].T \
    #     #            @ Atdup @ v0up @ VTup[filterup, :]
    #     # vbardn_b = - np.linalg.pinv(Atddn @ VTdn[filterdn, :].T, rcond=rcond) @ Udn[:, filterdn] @ Udn[:, filterdn].T \
    #     #            @ Atddn @ v0dn @ VTdn[filterdn, :]
    #
    #     # 2
    #     vbarup_a = np.linalg.pinv(Atdup @ VTup[filterup, :].T, rcond=rcond_bar) @ btdup @ VTup[filterup, :]
    #     vbardn_a = np.linalg.pinv(Atddn @ VTdn[filterup, :].T, rcond=rcond_bar) @ btddn @ VTdn[filterup, :]
    #
    #     vbarup_b = (np.linalg.pinv(Atdup @ VTup[filterup, :].T, rcond=rcond_bar) @ Atdup @ v0up) @ VTup[filterup, :]
    #     vbardn_b = (np.linalg.pinv(Atddn @ VTdn[filterup, :].T, rcond=rcond_bar) @ Atddn @ v0dn) @ VTdn[filterup, :]
    #
    #     # Monitor Session
    #     print("|v0|:", np.sum(np.abs(self.vp_basis.to_grid(v0up))*self.vp_basis.w), np.linalg.norm(v0up),
    #           "|vbara|:", np.sum(np.abs(self.vp_basis.to_grid(vbarup_a))*self.vp_basis.w), np.linalg.norm(vbarup_a),
    #           "|vbarb|:", np.sum(np.abs(self.vp_basis.to_grid(vbarup_b))*self.vp_basis.w), np.linalg.norm(vbarup_b))
    #
    #     stemp = np.copy(sup)
    #     stemp[filterup] = 0.0
    #     A0up = Uup @ np.diag(stemp) @ VTup
    #     print(np.linalg.norm(A0up @ btdup), np.linalg.norm(A0up @ v0up - b0up), np.linalg.norm(A0up @ vbarup_a))
    #
    #     vbarup = vbarup_a + vbarup_b
    #     vbardn = vbardn_a + vbardn_b
    #     # vbarup = vbarup_a
    #     # vbardn = vbardn_a
    #
    #     v_bar = np.concatenate((vbarup, vbardn))
    #     return v0, v_bar, (Atdup, Atddn)
    #     # return v0, v_bar, (vbarrup, vbarrdn), (Atdup, Atddn)

    def get_dvp_GL12(self, grad, hess, rcond=None, rcond_bar_a=None, rcond_bar_b=None):
        """
        To get v0 and v_bar from Gidopoulos, Lathiotakis, 2012 PRA 85, 052508.
        :param grad:
        :param hess:
        :param rcond:
        :return:
        """

        nbf = int(grad.shape[0]/2)
        grad_up = grad[:nbf]
        grad_dn = grad[nbf:]
        hess_up = hess[:nbf, :nbf]
        hess_dn = hess[nbf:, nbf:]

        Uup, sup, VTup = np.linalg.svd(hess_up)
        Udn, sdn, VTdn = np.linalg.svd(hess_dn)

        f,ax = plt.subplots(1,1,dpi=200)
        ax.axvline(x=20, ls="--", lw=0.7, color='r')
        ax.axvline(x=30, ls="--", lw=0.7, color='r')
        ax.axvline(x=40, ls="--", lw=0.7, color='r')
        ax.axvline(x=50, ls="--", lw=0.7, color='r')
        ax.scatter(range(nbf), np.log10(sup), s=1)


        # f.show()
        # rcond = int(input("Enter svd cut index: "))
        if rcond is None:
            rcond = -1
        rcond = sup[rcond] / sup[0] * 0.999

        filterup = sup < np.max(sup) * rcond
        filterdn = sdn < np.max(sdn) * rcond

        b0up = Uup[:, np.bitwise_not(filterup)] @ VTup[np.bitwise_not(filterup), :] @ grad_up
        b0dn = Udn[:, np.bitwise_not(filterdn)] @ VTdn[np.bitwise_not(filterdn), :] @ grad_dn
        btdup = grad_up - b0up
        btddn = grad_dn - b0dn

        sinv0up = 1/sup
        sinv0dn = 1/sdn
        sinv0up[filterup] = 0.0
        sinv0dn[filterdn] = 0.0

        v0up = VTup.T @ np.diag(sinv0up) @ Uup.T @ b0up
        v0dn = VTdn.T @ np.diag(sinv0dn) @ Udn.T @ b0dn
        # v0up = VTup.T @ np.diag(sinv0up) @ Uup.T @ grad_up
        # v0dn = VTdn.T @ np.diag(sinv0dn) @ Udn.T @ grad_dn

        v0 = np.concatenate((v0up, v0dn))

        if self.four_overlap_two_basis is None:
            self.four_overlap_two_basis = pdft.fouroverlap([self.molecule.wfn, self.vp_basis.wfn],
                                                 self.molecule.geometry, None, self.molecule.mints)[0]

        A_full_up = np.einsum("ijuv,im,jm->uv", self.four_overlap_two_basis,
                              self.molecule.Cocca.np, self.molecule.Cocca.np, optimize=True)
        A_full_dn = np.einsum("ijuv,im,jm->uv", self.four_overlap_two_basis,
                              self.molecule.Coccb.np, self.molecule.Coccb.np, optimize=True)

        Atdup = A_full_up - np.einsum('ai,bj,ci,dj,abm,cdn -> mn',
                                     self.molecule.Ca.np[:, :self.molecule.nalpha],
                                     self.molecule.Ca.np,
                                     self.molecule.Ca.np[:, :self.molecule.nalpha],
                                     self.molecule.Ca.np,
                                     self.three_overlap,
                                     self.three_overlap, optimize=True)

        Atddn = A_full_dn - np.einsum('ai,bj,ci,dj,abm,cdn -> mn',
                                     self.molecule.Cb.np[:, :self.molecule.nbeta],
                                     self.molecule.Cb.np,
                                     self.molecule.Cb.np[:, :self.molecule.nbeta],
                                     self.molecule.Cb.np,
                                     self.three_overlap,
                                     self.three_overlap, optimize=True)

        A0tempup = np.einsum('ai,bj,ci,dj,abm,cdn -> mn',
                              self.molecule.Ca.np[:, :self.molecule.nalpha],
                              self.molecule.Ca.np[:, self.molecule.nalpha:],
                              self.molecule.Ca.np[:, :self.molecule.nalpha],
                              self.molecule.Ca.np[:, self.molecule.nalpha:],
                              self.three_overlap,
                              self.three_overlap, optimize=True)
        A0tempdn = np.einsum('ai,bj,ci,dj,abm,cdn -> mn',
                             self.molecule.Cb.np[:, :self.molecule.nbeta],
                             self.molecule.Cb.np[:, self.molecule.nbeta:],
                             self.molecule.Cb.np[:, :self.molecule.nbeta],
                             self.molecule.Cb.np[:, self.molecule.nbeta:],
                             self.three_overlap,
                             self.three_overlap, optimize=True)

        # Rescaling
        sfullup = np.linalg.svd(A0tempup)[1]
        sfulldn = np.linalg.svd(A0tempdn)[1]

        # print(repr(sup), "\n", repr(sfullup))
        # Atdup *= sup[0] / sfullup[0]
        # Atddn *= sdn[0] / sfulldn[0]

        #
        Unullup = Uup[:, filterup]
        Unulldn = Udn[:, filterdn]
        Vnullup = VTup[filterup, :].T
        Vnulldn = VTdn[filterdn, :].T

        Xuvup = Unullup.T @ Atdup @ Vnullup
        Xuvdn = Unulldn.T @ Atddn @ Vnulldn

        stemp = np.linalg.svd(Xuvup)[1]
        ax.scatter(range(stemp.shape[0]), np.log10(stemp), s=1)
        # rcond_bar = int(input("Enter svd cut index: "))


        Utdup, stdup, VTtdup = np.linalg.svd(Atdup)
        Utddn, stddn, VTtddn = np.linalg.svd(Atddn)
        stdup_temp = np.copy(stdup)
        stddn_temp = np.copy(stddn)

        if rcond_bar_b is None and rcond_bar_a is not None:
            rcond_bar_b = rcond_bar_a
        elif rcond_bar_a is None and rcond_bar_b is not None:
            rcond_bar_a = rcond_bar_b
        elif rcond_bar_a is None and rcond_bar_b is None:
            rcond_bar_a = -1
            rcond_bar_b = -1

        # f.show()
        # rcond_bar_a = int(input("Enter svd cut index: "))
        # rcond_bar_b = int(input("Enter svd cut index: "))
        ax.axvline(x=stemp.shape[0] + rcond_bar_a, ls=":", lw=0.7, color='b')
        ax.axvline(x=stemp.shape[0] + rcond_bar_b, ls="-.", lw=0.7, color='b')
        # f.show()
        plt.close(f)
        print(repr(stemp))

        # stdup_temp[stdup<stdup[rcond_bar_b]] = 0
        # stddn_temp[stddn<stddn[rcond_bar_b]] = 0
        rcond_bar_b = stemp[rcond_bar_b] / stemp[0] * 0.999
        rcond_bar_a = stemp[rcond_bar_a] / stemp[0] * 0.999


        # # vbar
        # 2
        # vbarup_a = Vnullup @ (np.linalg.pinv(Xuvup, rcond=rcond_bar_a) @ Unullup.T @ btdup)
        # vbarup_b = - Vnullup @ (np.linalg.pinv(Xuvup, rcond=rcond_bar_b) @ Unullup.T @ (Utdup @ np.diag(stdup_temp) @ VTtdup) @ v0up)
        # vbardn_a = Vnulldn @ (np.linalg.pinv(Xuvdn, rcond=rcond_bar_a) @ Unulldn.T @ btddn)
        # vbardn_b = - Vnulldn @ (np.linalg.pinv(Xuvdn, rcond=rcond_bar_b) @ Unulldn.T @ (Utddn @ np.diag(stddn_temp) @ VTtddn) @ v0dn)

        # 4
        vbarup_a = Vnullup @ (np.linalg.pinv(Xuvup, rcond=rcond_bar_a) @ Unullup.T @ btdup)
        vbarup_b = - Vnullup @ (np.linalg.pinv(Xuvup, rcond=rcond_bar_b) @ Unullup.T @ Atdup @ v0up)
        vbardn_a = Vnulldn @ (np.linalg.pinv(Xuvdn, rcond=rcond_bar_a) @ Unulldn.T @ btddn)
        vbardn_b = - Vnulldn @ (np.linalg.pinv(Xuvdn, rcond=rcond_bar_b) @ Unulldn.T @ Atddn @ v0dn)

        # vbarup_a += Vnullup @ pdft.inv_pinv(Xuvup, 21, 30) @ Vnullup.T @ btdup
        # vbarup_b += - Vnullup @ pdft.inv_pinv(Xuvup, 21, 30) @ Vnullup.T @ Atdup @ v0up
        # vbardn_a += Vnulldn @ pdft.inv_pinv(Xuvdn, 21, 30) @ Vnulldn.T @ btddn
        # vbardn_b += - Vnulldn @ pdft.inv_pinv(Xuvdn, 21, 30) @ Vnulldn.T @ Atddn @ v0dn

        # 1
        # vbarup_a = Vnullup @ np.linalg.solve(Xuvup, Vnullup.T @ btdup)
        # vbarup_b = - Vnullup @ np.linalg.solve(Xuvup, Vnullup.T @ Atdup @ v0up)
        # vbardn_a = Vnulldn @ np.linalg.solve(Xuvdn, Vnulldn.T @ btddn)
        # vbardn_b = - Vnulldn @ np.linalg.solve(Xuvdn, Vnulldn.T @ Atddn @ v0dn)

        # 3
        # vbarup_a = Unullup @ np.linalg.solve(Xuvup, Unullup.T @ btdup)
        # vbarup_b = - Unullup @ Unullup.T @ v0up
        # vbardn_a = Unulldn @ np.linalg.solve(Xuvdn, Unulldn.T @ btddn)
        # vbardn_b = - Unulldn @ Unullup.T @ v0dn

        # Monitor Session
        print("|v0|:", np.sum(np.abs(self.vp_basis.to_grid(v0up))*self.vp_basis.w), np.linalg.norm(v0up),
              "|vbara|:", np.sum(np.abs(self.vp_basis.to_grid(vbarup_a))*self.vp_basis.w), np.linalg.norm(vbarup_a),
              "|vbarb|:", np.sum(np.abs(self.vp_basis.to_grid(vbarup_b))*self.vp_basis.w), np.linalg.norm(vbarup_b))

        # Rescaling again
        # vbarup_a *= np.sum(np.abs(self.vp_basis.to_grid(v0up))*self.vp_basis.w) / np.sum(np.abs(self.vp_basis.to_grid(vbarup_a))*self.vp_basis.w)
        # vbardn_a *= np.sum(np.abs(self.vp_basis.to_grid(v0up))*self.vp_basis.w) / np.sum(np.abs(self.vp_basis.to_grid(vbarup_a))*self.vp_basis.w)
        # vbarup_a *= np.linalg.norm(v0up) / np.linalg.norm(vbarup_a)
        # vbardn_a *= np.linalg.norm(v0dn) / np.linalg.norm(vbardn_a)
        # vbarup_b *= np.linalg.norm(v0up) / np.linalg.norm(vbarup_b)
        # vbardn_b *= np.linalg.norm(v0dn) / np.linalg.norm(vbardn_b)

        stemp = np.copy(sup)
        stemp[filterup] = 0.0
        A0up = Uup @ np.diag(stemp) @ VTup
        print(np.linalg.norm(A0up @ btdup),
              np.linalg.norm(A0up @ v0up - b0up),
              np.linalg.norm(A0up @ vbarup_a))
        stemp = np.linalg.svd(Xuvup)[1]
        print(stemp[0]/stemp[-1])

        # vbarup = vbarup_a + vbarup_b
        # vbardn = vbardn_a + vbardn_b
        vbarup = vbarup_a
        vbardn = vbardn_a

        v_bar = np.concatenate((vbarup, vbardn))
        return v0, v_bar, np.concatenate((vbarup_b, vbardn_b))

    def get_dvp_GL12_eig_expended(self, grad, hess, rcond=None, rcond_bar_a=None, rcond_bar_b=None):
        """
        To get v0 and v_bar from Gidopoulos, Lathiotakis, 2012 PRA 85, 052508.
        :param grad:
        :param hess:
        :param rcond:
        :return:
        """

        nbf = int(grad.shape[0]/2)
        grad_up = grad[:nbf]
        grad_dn = grad[nbf:]
        hess_up = hess[:nbf, :nbf]
        hess_dn = hess[nbf:, nbf:]

        # Not sure why but scipy.linalg.eig does not seem to work for this.
        A = self.vp_basis.A.np
        S = self.vp_basis.S.np

        hess_up_psi4 = psi4.core.Matrix.from_array(A @ hess_up @ A)
        Cup_CE = psi4.core.Matrix(nbf, nbf)
        sup = psi4.core.Vector(nbf)
        hess_up_psi4.diagonalize(Cup_CE, sup, psi4.core.DiagonalizeOrder.Descending)
        Cup_CE = A @ Cup_CE.np
        sup = sup.np

        assert np.allclose(Cup_CE.T @ S @ Cup_CE, np.eye(nbf))

        hess_dn_psi4 = psi4.core.Matrix.from_array(A @ hess_dn @ A)
        Cdn_CE = psi4.core.Matrix(nbf, nbf)
        sdn = psi4.core.Vector(nbf)
        hess_dn_psi4.diagonalize(Cdn_CE, sdn, psi4.core.DiagonalizeOrder.Descending)
        Cdn_CE = A @ Cdn_CE.np
        sdn = sdn.np

        f,ax = plt.subplots(1,1,dpi=200)
        ax.axvline(x=20, ls="--", lw=0.7, color='r')
        ax.axvline(x=30, ls="--", lw=0.7, color='r')
        ax.axvline(x=40, ls="--", lw=0.7, color='r')
        ax.axvline(x=50, ls="--", lw=0.7, color='r')
        ax.scatter(range(nbf), np.log10(np.abs(sup)), s=1)

        if rcond is None:
            rcond = -1
        rcond = sup[rcond] / sup[0] * 0.999

        # f.show()
        # rcond = int(input("Enter svd cut index: "))

        filterup = np.abs(sup) < (np.max(np.abs(sup)) * rcond)
        filterdn = np.abs(sdn) < (np.max(np.abs(sdn)) * rcond)

        b0up_CE = Cup_CE[:,np.bitwise_not(filterup)] @ Cup_CE[:,np.bitwise_not(filterup)].T @ grad_up
        b0dn_CE = Cdn_CE[:,np.bitwise_not(filterdn)] @ Cdn_CE[:,np.bitwise_not(filterdn)].T @ grad_dn
        b0up = S @ b0up_CE
        b0dn = S @ b0dn_CE
        btdup = grad_up - b0up
        btddn = grad_dn - b0up

        sinv0up = 1/sup
        sinv0dn = 1/sdn
        sinv0up[filterup] = 0.0
        sinv0dn[filterdn] = 0.0

        v0up_CE = Cup_CE @ np.diag(sinv0up) @ Cup_CE.T @ b0up
        v0dn_CE = Cdn_CE @ np.diag(sinv0dn) @ Cdn_CE.T @ b0dn

        v0_CE = np.concatenate((v0up_CE, v0dn_CE))

        if self.four_overlap_two_basis is None:
            self.four_overlap_two_basis = pdft.fouroverlap([self.molecule.wfn, self.vp_basis.wfn],
                                                 self.molecule.geometry, None, self.molecule.mints)[0]

        A_full_up = np.einsum("ijuv,im,jm->uv", self.four_overlap_two_basis,
                              self.molecule.Cocca.np, self.molecule.Cocca.np, optimize=True)
        A_full_dn = np.einsum("ijuv,im,jm->uv", self.four_overlap_two_basis,
                              self.molecule.Coccb.np, self.molecule.Coccb.np, optimize=True)

        Atdup = A_full_up - np.einsum('ai,bj,ci,dj,abm,cdn -> mn',
                                     self.molecule.Ca.np[:, :self.molecule.nalpha],
                                     self.molecule.Ca.np,
                                     self.molecule.Ca.np[:, :self.molecule.nalpha],
                                     self.molecule.Ca.np,
                                     self.three_overlap,
                                     self.three_overlap, optimize=True)

        Atddn = A_full_dn - np.einsum('ai,bj,ci,dj,abm,cdn -> mn',
                                     self.molecule.Cb.np[:, :self.molecule.nbeta],
                                     self.molecule.Cb.np,
                                     self.molecule.Cb.np[:, :self.molecule.nbeta],
                                     self.molecule.Cb.np,
                                     self.three_overlap,
                                     self.three_overlap, optimize=True)

        A0tempup = np.einsum('ai,bj,ci,dj,abm,cdn -> mn',
                              self.molecule.Ca.np[:, :self.molecule.nalpha],
                              self.molecule.Ca.np[:, self.molecule.nalpha:],
                              self.molecule.Ca.np[:, :self.molecule.nalpha],
                              self.molecule.Ca.np[:, self.molecule.nalpha:],
                              self.three_overlap,
                              self.three_overlap, optimize=True)
        A0tempdn = np.einsum('ai,bj,ci,dj,abm,cdn -> mn',
                             self.molecule.Cb.np[:, :self.molecule.nbeta],
                             self.molecule.Cb.np[:, self.molecule.nbeta:],
                             self.molecule.Cb.np[:, :self.molecule.nbeta],
                             self.molecule.Cb.np[:, self.molecule.nbeta:],
                             self.three_overlap,
                             self.three_overlap, optimize=True)

        # Rescaling
        sfullup = np.linalg.svd(A0tempup)[1]
        sfulldn = np.linalg.svd(A0tempdn)[1]

        # print(repr(sup), "\n", repr(sfullup))
        # Atdup *= sup[0] / sfullup[0]
        # Atddn *= sdn[0] / sfulldn[0]

        #
        Cnullup = Cup_CE[:, filterup]
        Cnulldn = Cdn_CE[:, filterdn]

        Xuvup = Cnullup.T @ Atdup @ Cnullup
        Xuvdn = Cnulldn.T @ Atddn @ Cnulldn

        stemp = np.linalg.svd(Xuvup)[1]
        ax.scatter(range(stemp.shape[0]), np.log10(np.abs(stemp)), s=1)

        Utdup, stdup, VTtdup = np.linalg.svd(Atdup)
        Utddn, stddn, VTtddn = np.linalg.svd(Atddn)
        stdup_temp = np.copy(stdup)
        stddn_temp = np.copy(stddn)

        if rcond_bar_b is None and rcond_bar_a is not None:
            rcond_bar_b = rcond_bar_a
        elif rcond_bar_a is None and rcond_bar_b is not None:
            rcond_bar_a = rcond_bar_b
        elif rcond_bar_a is None and rcond_bar_b is None:
            rcond_bar_a = -1
            rcond_bar_b = -1

        # f.show()
        # rcond_bar_a = int(input("Enter svd cut index: "))
        # rcond_bar_b = int(input("Enter svd cut index: "))
        ax.axvline(x=stemp.shape[0] + rcond_bar_a, ls=":", lw=0.7, color='b')
        ax.axvline(x=stemp.shape[0] + rcond_bar_b, ls="-.", lw=0.7, color='b')
        # f.show( )
        plt.close(f)
        print(repr(stemp))
        # stdup_temp[stdup<stdup[rcond_bar_b]] = 0
        # stddn_temp[stddn<stddn[rcond_bar_b]] = 0
        rcond_bar_b = stemp[rcond_bar_b] / stemp[0] * 0.999
        rcond_bar_a = stemp[rcond_bar_a] / stemp[0] * 0.999

        # # vbar
        # 2
        # vbarup_a_CE = Cnullup @ (np.linalg.pinv(Xuvup, rcond=rcond_bar_a) @ Cnullup.T @ btdup)
        # vbarup_b_CE = - Cnullup @ (np.linalg.pinv(Xuvup, rcond=rcond_bar_b) @ Cnullup.T @ (Utdup @ np.diag(stdup_temp) @ VTtdup) @ v0up_CE)
        # vbardn_a_CE = Cnulldn @ (np.linalg.pinv(Xuvdn, rcond=rcond_bar_a) @ Cnulldn.T @ btddn)
        # vbardn_b_CE = - Cnulldn @ (np.linalg.pinv(Xuvdn, rcond=rcond_bar_b) @ Cnulldn.T @ (Utddn @ np.diag(stddn_temp) @ VTtddn) @ v0dn_CE)

        # 4
        vbarup_a_CE = Cnullup @ (np.linalg.pinv(Xuvup, rcond=rcond_bar_a) @ Cnullup.T @ btdup)
        vbarup_b_CE = - Cnullup @ (np.linalg.pinv(Xuvup, rcond=rcond_bar_b) @ Cnullup.T @ Atdup @ v0up_CE)
        vbardn_a_CE = Cnulldn @ (np.linalg.pinv(Xuvdn, rcond=rcond_bar_a) @ Cnulldn.T @ btddn)
        vbardn_b_CE = - Cnulldn @ (np.linalg.pinv(Xuvdn, rcond=rcond_bar_b) @ Cnulldn.T @ Atddn @ v0dn_CE)

        # vbarup_a_CE += Vnullup @ pdft.inv_pinv(Xuvup, 21, 30) @ Vnullup.T @ btdup
        # vbarup_b_CE += - Vnullup @ pdft.inv_pinv(Xuvup, 21, 30) @ Vnullup.T @ Atdup @ v0up_CE
        # vbardn_a_CE += Vnulldn @ pdft.inv_pinv(Xuvdn, 21, 30) @ Vnulldn.T @ btddn
        # vbardn_b_CE += - Vnulldn @ pdft.inv_pinv(Xuvdn, 21, 30) @ Vnulldn.T @ Atddn @ v0dn_CE

        # 1
        # vbarup_a_CE = Cnullup @ np.linalg.solve(Xuvup, Cnullup.T @ btdup)
        # vbarup_b_CE = - Cnullup @ np.linalg.solve(Xuvup, Cnullup.T @ Atdup @ v0up_CE)
        # vbardn_a_CE = Cnulldn @ np.linalg.solve(Xuvdn, Cnulldn.T @ btddn)
        # vbardn_b_CE = - Cnulldn @ np.linalg.solve(Xuvdn,  Cnulldn.T @ Atddn @ v0dn_CE)

        # 3
        # vbarup_a_CE = Unullup @ np.linalg.solve(Xuvup, Unullup.T @ btdup)
        # vbarup_b_CE = - Unullup @ Unullup.T @ v0up_CE
        # vbardn_a_CE = Unulldn @ np.linalg.solve(Xuvdn, Unulldn.T @ btddn)
        # vbardn_b_CE = - Unulldn @ Unullup.T @ v0dn_CE

        # Monitor Session
        print("|v0|:", np.sum(np.abs(self.vp_basis.to_grid(v0up_CE))*self.vp_basis.w), np.linalg.norm(v0up_CE),
              "|vbara|:", np.sum(np.abs(self.vp_basis.to_grid(vbarup_a_CE))*self.vp_basis.w), np.linalg.norm(vbarup_a_CE),
              "|vbarb|:", np.sum(np.abs(self.vp_basis.to_grid(vbarup_b_CE))*self.vp_basis.w), np.linalg.norm(vbarup_b_CE))

        # Rescaling again
        # vbarup_a_CE *= np.sum(np.abs(self.vp_basis.to_grid(v0up_CE))*self.vp_basis.w) / np.sum(np.abs(self.vp_basis.to_grid(vbarup_a_CE))*self.vp_basis.w)
        # vbardn_a_CE *= np.sum(np.abs(self.vp_basis.to_grid(v0up_CE))*self.vp_basis.w) / np.sum(np.abs(self.vp_basis.to_grid(vbarup_a_CE))*self.vp_basis.w)
        # vbarup_a_CE *= np.linalg.norm(v0up_CE) / np.linalg.norm(vbarup_a_CE)
        # vbardn_a_CE *= np.linalg.norm(v0dn_CE) / np.linalg.norm(vbardn_a_CE)
        # vbarup_b_CE *= np.linalg.norm(v0up_CE) / np.linalg.norm(vbarup_b_CE)
        # vbardn_b_CE *= np.linalg.norm(v0dn_CE) / np.linalg.norm(vbardn_b_CE)

        stemp = np.copy(sup)
        stemp[filterup] = 0.0
        print(np.linalg.norm(Cup_CE @ np.diag(stemp) @ Cup_CE.T @ btdup),
              np.linalg.norm(Cup_CE @ np.diag(stemp) @ Cup_CE.T @ S @ v0up_CE - b0up_CE),
              np.linalg.norm(Cup_CE @ np.diag(stemp) @ Cup_CE.T @ S @ vbarup_a_CE),
              np.linalg.norm(Cup_CE @ np.diag(stemp) @ Cup_CE.T @ S @ vbarup_b_CE))
        stemp = np.linalg.svd(Xuvup)[1]
        print(stemp[0]/stemp[-1])

        # vbarup = vbarup_a_CE + vbarup_b_CE
        # vbardn = vbardn_a_CE + vbardn_b_CE
        vbarup_CE = vbarup_a_CE
        vbardn_CE = vbardn_a_CE

        v_bar_CE = np.concatenate((vbarup_CE, vbardn_CE))
        return v0_CE, v_bar_CE, np.concatenate((vbarup_b_CE, vbardn_b_CE))

    def find_vxc_manualNewton(self, maxiter=49, svd_rcond=None, c1=1e-4, c2=0.99, c3=1e-2,
                              svd_parameter=None, line_search_method="StrongWolfe",
                              BT_beta_threshold=1e-7, rho_conv_threshold=1e-3,
                              continue_opt=False, find_vxc_grid=True,
                              grad_norm=1e-2):
        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        if not continue_opt:
            print("Zero the old result for a new calculation..")
            self.v_output = np.zeros_like(self.v_output)
            self.v0_output = np.zeros_like(self.v_output)
            self.vbara_output = np.zeros_like(self.v_output)
            self.vbarb_output = np.zeros_like(self.v_output)

            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.KS_solver(100, [Vks_a, Vks_b])
            self.update_vout_constant()

        ## Tracking rho and changing beta
        n = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
        n_input = self.molecule.to_grid(self.input_density_wfn.Da().np+self.input_density_wfn.Db().np)
        dn_before = np.sum(np.abs(n - n_input) * self.molecule.w)
        L_old = self.Lagrangian_WuYang()

        print("Initial dn:", dn_before, "Initial L:", L_old)

        LineSearch_converge_flag = False
        beta = 2

        ls = 0  # length segment list
        ns = 0  # current pointer on the segment list

        cycle_n = np.inf  # Density of last segment cycle
        if line_search_method == "StrongWolfeD":
            def density_improvement_test(alpha, x, f, g):
                #  alpha, x, f, g in principle are not needed.
                n = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
                dn = np.sum(np.abs(n - n_input) * self.molecule.w)
                if not dn - dn_before < - c3 * dn_before:
                    print("Density improvement?", dn - dn_before, - c3 * dn_before)
                return dn - dn_before < - c3 * dn_before

        print("<<<<<<<<<<<<<<<<<<<<<<WuYang vxc Inversion manual Newton<<<<<<<<<<<<<<<<<<<")
        self.svd_index = svd_rcond
        for scf_step in range(1, maxiter + 1):

            # if scf_step == 2:
            #     self.update_v0()

            hess = self.hess_WuYang(v=self.v_output)
            jac = self.grad_WuYang(v=self.v_output)

            if np.linalg.norm(jac) <= grad_norm:
                break

            if svd_rcond is None:
                dv = -np.linalg.solve(hess, jac)
            # The svd pseudo-inverse could hopefully be avoided, with orthogonal v_basis.
            elif type(svd_rcond) is float:
                hess_inv = np.linalg.pinv(hess, rcond=svd_rcond)
                dv = -np.dot(hess_inv, jac)
            elif type(svd_rcond) is int:
                s = np.linalg.svd(hess)[1]
                self.svd_index = svd_rcond
                svd = s[self.svd_index]
                # print(svd)
                svd = svd * 0.999 / s[0]
                hess_inv = np.linalg.pinv(hess, rcond=svd)
                dv = -np.dot(hess_inv, jac)
            elif svd_rcond == "GL":
                v0, v_bara, v_barb = self.get_dvp_GL12_eig_expended(jac, hess, rcond=svd_parameter[0],
                                                                    rcond_bar_a=svd_parameter[1],
                                                                    rcond_bar_b=svd_parameter[2])
                # nbf = int(jac.shape[0]/2)
                # vup = - np.linalg.solve(hess[:nbf, :nbf] + Aup, jac[:nbf])
                # vdn = - np.linalg.solve(hess[nbf:, nbf:] + Adn, jac[nbf:])

                # vup = - np.linalg.pinv(hess[:nbf, :nbf] + Aup, rcond=1e-4) @ jac[:nbf]
                # vdn = - np.linalg.pinv(hess[nbf:, nbf:] + Adn, rcond=1e-4) @ jac[nbf:]
                # dv = np.concatenate((vup, vdn))

                # if scf_step % 3 == 0:
                #     dv = -v0
                # elif scf_step % 2 == 0:
                #     dv = -v_bara
                # else:
                #     dv = -v_barb

                dv = - (v0 + v_bara + v_barb)
                # dv = - v0
            elif svd_rcond == "GL_mod":
                v0, v_bara, v_barb = self.get_dvp_GL12(jac, hess, rcond=svd_parameter[0],
                                                        rcond_bar_a=svd_parameter[1],
                                                        rcond_bar_b=svd_parameter[2])
                # nbf = int(jac.shape[0]/2)
                # vup = - np.linalg.solve(hess[:nbf, :nbf] + Aup, jac[:nbf])
                # vdn = - np.linalg.solve(hess[nbf:, nbf:] + Adn, jac[nbf:])

                # vup = - np.linalg.pinv(hess[:nbf, :nbf] + Aup, rcond=1e-4) @ jac[:nbf]
                # vdn = - np.linalg.pinv(hess[nbf:, nbf:] + Adn, rcond=1e-4) @ jac[nbf:]
                # dv = np.concatenate((vup, vdn))

                # if scf_step % 3 == 0:
                #     dv = -v0
                # elif scf_step % 2 == 0:
                #     dv = -v_bara
                # else:
                #     dv = -v_barb

                dv = - (v0 + v_bara + v_barb)
                # dv = - v0
            elif svd_rcond == "input_once":
                s = np.linalg.svd(hess)[1]

                if scf_step == 1:
                    print(repr(s))

                    plt.figure(dpi=200)
                    # plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.title(str(self.ortho_basis))
                    plt.show()
                    plt.close()

                    self.svd_index = int(input("Enter svd cut index (0-based indexing): "))
                svd = s[self.svd_index]
                # print(svd)
                svd = svd * 0.9999 / s[0]

                hess_inv = np.linalg.pinv(hess, rcond=svd)
                dv = -np.dot(hess_inv, jac)
            elif svd_rcond == "gradient_descent":
                dv = - jac
            elif svd_rcond == "input_every":
                s = np.linalg.svd(hess)[1]

                # print(repr(s))

                plt.figure(dpi=200)
                # plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                plt.title(str(self.ortho_basis))
                plt.show()
                plt.close()

                self.svd_index = int(input("Enter svd cut index (0-based indexing): "))
                svd = s[self.svd_index]
                # print(svd)
                svd = svd * 0.95 / s[0]

                hess_inv = np.linalg.pinv(hess, rcond=svd)
                dv = -np.dot(hess_inv, jac)

            elif svd_rcond == "increase":
                if svd_parameter is None:
                    svd_parameter = [0, 10] # [Starting percentile, update step ]
                s = np.linalg.svd(hess)[1]

                if scf_step == 1:
                    print(repr(s))

                    plt.figure(dpi=200)
                    # plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.title(str(self.ortho_basis))
                    plt.show()
                    plt.close()

                    end = int(input("Enter svd_rcond end number: "))
                    svd_parameter.append(end)

                if scf_step == 1:
                    self.svd_index = int(s.shape[0]*svd_parameter[0])
                elif beta <= BT_beta_threshold:
                    # SVD move on
                    self.svd_index += int(svd_parameter[1])

                if self.svd_index >= svd_parameter[-1]:
                    self.svd_index = svd_parameter[-1]
                    LineSearch_converge_flag = True

                svd = s[self.svd_index-1]
                # print(svd)
                svd = svd * 0.95 / s[0]

                hess_inv = np.linalg.pinv(hess, rcond=svd)
                dv = -np.dot(hess_inv, jac)

            elif svd_rcond == "cycle_segment":
                if svd_parameter is None:
                    # [seg step length, current start idx]
                    svd_parameter = [0.1, 0]

                s = np.linalg.svd(hess)[1]
                # SVD move on
                s_shape = s.shape[0]

                if scf_step == 1:
                    plt.figure(dpi=200)
                    # plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.title(str(self.ortho_basis))
                    plt.show()
                    plt.close()

                    end = int(input("Enter svd_rcond end number: "))
                    svd_parameter.append(end)

                # another cycle
                elif beta <= BT_beta_threshold:
                    if svd_parameter[-1] == -1:
                        if svd_parameter[1] >= s_shape:
                            svd_parameter[1] = 0
                    else:
                        assert svd_parameter[-1] > 0
                        if svd_parameter[1] >= svd_parameter[-1]:
                            svd_parameter[1] = 0

                    start = svd_parameter[1]
                    svd_parameter[1] += int(svd_parameter[0]*s_shape)

                    if svd_parameter[-1] == -1:
                        if svd_parameter[1] > s_shape:
                            svd_parameter[1] = s_shape
                    else:
                        assert svd_parameter[-1] > 0
                        if svd_parameter[1] > svd_parameter[-1]:
                            svd_parameter[1] = svd_parameter[-1]

                    end = svd_parameter[1]

                self.svd_index = [start, end]

                hess_inv = pdft.inv_pinv(hess, self.svd_index[0], self.svd_index[1])
                dv = -np.dot(hess_inv, jac)

            elif svd_rcond == "segment_cycle":
                if svd_parameter is None:
                    # [seg step length, current start idx]
                    svd_parameter = [0.1, 0]

                s = np.linalg.svd(hess)[1]
                # SVD move on
                s_shape = s.shape[0]

                if scf_step == 1:
                    print(repr(s))

                    plt.figure(dpi=200)
                    # plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.scatter(range(s.shape[0]), np.log10(s), s=1)
                    plt.title(str(self.ortho_basis))
                    plt.show()
                    plt.close()

                    end = int(input("Enter svd_rcond end number: "))
                    svd_parameter.append(end)

                # another cycle
                if svd_parameter[-1] == -1:
                    if svd_parameter[1] >= s_shape:
                        svd_parameter[1] = 0
                else:
                    assert svd_parameter[-1] > 0
                    if svd_parameter[1] >= svd_parameter[-1]:
                        svd_parameter[1] = 0

                start = svd_parameter[1]
                svd_parameter[1] += int(svd_parameter[0]*s_shape)

                if svd_parameter[-1] == -1:
                    if svd_parameter[1] > s_shape:
                        svd_parameter[1] = s_shape
                else:
                    assert svd_parameter[-1] > 0
                    if svd_parameter[1] > svd_parameter[-1]:
                        svd_parameter[1] = svd_parameter[-1]

                end = svd_parameter[1]

                self.svd_index = [start, end]

                hess_inv = pdft.inv_pinv(hess, self.svd_index[0], self.svd_index[1])

                dv = -np.dot(hess_inv, jac)

            elif svd_rcond == "input_segment_cycle":
                s = np.linalg.svd(hess)[1]
                # SVD move on
                s_shape = s.shape[0]

                if scf_step == 1:
                    print(repr(s))

                    plt.figure(dpi=200)
                    # plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.scatter(range(s.shape[0]), np.log10(s), s=1)
                    plt.title(str(self.ortho_basis))
                    plt.show()
                    plt.close()

                    # Get the segment partation.
                    svd_parameter = [0]
                    ns = int(input("Enter number of segments: "))
                    for i in range(0, ns):
                        ele = int(input()) + 1
                        svd_parameter.append(ele)  # adding the element

                # Cycle number
                # ns=4
                # [0-10,10-20,20-30,30-40]
                # svd_parameter = [0,10,20,30,40]
                # start = [0, 10, 20, 30]
                # end = [10, 20, 30, 40]
                # cn = [1, 2, 3, 0]
                cn = scf_step % ns

                start = svd_parameter[:-1][cn-1]
                end = svd_parameter[1:][cn-1]

                self.svd_index = [start, end]

                hess_inv = pdft.inv_pinv(hess, self.svd_index[0], self.svd_index[1])

                dv = -np.dot(hess_inv, jac)

            elif svd_rcond == "search_segment_cycle":
                # Segment move on in each iter
                s = np.linalg.svd(hess)[1]
                # SVD move on
                s_shape = s.shape[0]

                if svd_parameter is None:
                    svd_parameter = 10  # the cutoff threshold.

                # Segmentation
                if scf_step==1 or ns==ls:
                    if np.sum(np.abs(cycle_n-n)*self.molecule.w) < rho_conv_threshold:
                        print("Break because n is not improved in this segment cycle.",
                              np.sum(np.abs(cycle_n-n)*self.molecule.w))
                        break
                    cycle_n = n
                    seg_list = [0]
                    for i in range(1,s_shape):
                        if s[i-1]/s[i] > svd_parameter:
                            # print(s[i], s[i-1])
                            seg_list.append(i)
                    seg_list.append(s_shape)
                    ls = len(seg_list) - 1  # length of segments
                    ns = 0  # segment number starts from 0
                    print("\nSegment", seg_list)
                    # print("\n")
                start = seg_list[ns]
                end = seg_list[ns+1]

                self.svd_index = [start, end]

                hess_inv = pdft.inv_pinv(hess, self.svd_index[0], self.svd_index[1])

                dv = -np.dot(hess_inv, jac)

                ns += 1

            elif svd_rcond == "segment_cycle_cutoff":
                # Segment move on in each iter
                s = np.linalg.svd(hess)[1]
                # SVD move on
                s_shape = s.shape[0]

                if svd_parameter is None:
                    svd_parameter = [3]  # the cutoff threshold.
                    print("TSVE segment cutoff ratio is chosen to be: ", svd_parameter[0])

                if scf_step==1:
                    print(repr(s))

                    plt.figure(dpi=200)
                    # plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.title(str(self.ortho_basis))
                    plt.show()
                    plt.close()

                    svd_cutoff = int(input("Enter svd cut index (0-based indexing): "))
                    svd_parameter.append(svd_cutoff)

                # Segmentation
                if scf_step==1 or ns==ls:
                    if np.sum(np.abs(cycle_n-n)*self.molecule.w) < rho_conv_threshold:
                        print("Break because n is not improved in this segment cycle.",
                              np.sum(np.abs(cycle_n-n)*self.molecule.w))
                        break
                    cycle_n = n
                    seg_list = [0]
                    # for i in range(1, s_shape):
                    #     if s[i-1]/s[i] > svd_parameter[0]:
                    #         # print(s[i], s[i-1])
                    #         seg_list.append(i)
                    #     if i==svd_parameter[-1]+1:
                    #         seg_list.append(svd_parameter[-1]+1)
                    # seg_list.append(s_shape)
                    for i in range(1, svd_parameter[-1]+1):
                        if s[i-1]/s[i] > svd_parameter[0]:
                            # print(s[i], s[i-1])
                            seg_list.append(i)
                    seg_list.append(svd_parameter[-1]+1)
                    ls = len(seg_list) - 1  # length of segments
                    ns = 0  # segment number starts from 0
                    print("\nSegment", seg_list)
                    # print("\n")
                start = seg_list[ns]
                end = seg_list[ns+1]

                self.svd_index = [start, end]

                hess_inv = pdft.inv_pinv(hess, self.svd_index[0], self.svd_index[1])

                dv = -np.dot(hess_inv, jac)

                ns += 1

            elif svd_rcond == "search_cycle_segment":
                # Segment move on when not improved
                s = np.linalg.svd(hess)[1]
                # SVD move on
                s_shape = s.shape[0]

                if svd_parameter is None:
                    svd_parameter = 10  # the cutoff threshold.

                if LineSearch_converge_flag:
                    ns += 1

                # Segmentation
                if scf_step==1 or ns==ls and (LineSearch_converge_flag):
                    if np.sum(np.abs(cycle_n-n)*self.molecule.w) < rho_conv_threshold:
                        print("Break because n is not improved in this segment cycle.", np.abs(cycle_n-n))
                        break
                    cycle_n = n
                    seg_list = [0]
                    for i in range(1,s_shape):
                        if s[i-1]/s[i] > svd_parameter:
                            # print(s[i], s[i-1])
                            seg_list.append(i)
                    seg_list.append(s_shape)
                    ls = len(seg_list) - 1  # length of segments
                    ns = 0  # segment number starts from 0
                    print("\nSegment", seg_list)
                    # print("\n")
                start = seg_list[ns]
                end = seg_list[ns+1]

                self.svd_index = [start, end]

                hess_inv = pdft.inv_pinv(hess, self.svd_index[0], self.svd_index[1])

                dv = -np.dot(hess_inv, jac)


            elif svd_rcond == "segments":
                if svd_parameter is None:
                    svd_parameter = [10, 0]
                s = np.linalg.svd(hess)[1]
                s_shape = s.shape[0]
                start = svd_parameter[1]
                svd_parameter[1] += int(s_shape / svd_parameter[0])
                if svd_parameter[1] > s_shape:
                    svd_parameter[1] = s_shape
                    LineSearch_converge_flag = True

                end = svd_parameter[1]

                self.svd_index = [start, end+1]

                hess_inv = pdft.inv_pinv(hess, start, end)
                dv = -np.dot(hess_inv, jac)

            if line_search_method == "BackTracking":
                beta = 2
                LineSearch_converge_flag = False
                while True:
                    beta *= 0.5
                    if beta < BT_beta_threshold:
                        LineSearch_converge_flag = True
                        break
                    # Traditional WuYang
                    v_temp = self.v_output + beta * dv
                    L = self.Lagrangian_WuYang(v=v_temp)
                    print(beta, L - L_old, c1 * beta * np.sum(jac * dv))
                    if L - L_old <= c1 * beta * np.sum(jac * dv) and beta * np.sum(jac * dv) < 0:
                        L_old = L
                        self.v_output = v_temp
                        n = self.molecule.to_grid(self.molecule.Da.np+self.molecule.Db.np)
                        dn_before = np.sum(np.abs(n_input - n) * self.molecule.w)
                        break
            elif line_search_method == "D":
                beta = 2
                LineSearch_converge_flag = False

                while True:
                    beta *= 0.5
                    if beta < BT_beta_threshold:
                        LineSearch_converge_flag = True
                        break
                    # Traditional WuYang
                    v_temp = self.v_output + beta * dv
                    L = self.Lagrangian_WuYang(v=v_temp)
                    n = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
                    dn = np.sum(np.abs(n - n_input) * self.molecule.w)
                    print(beta, L - L_old, dn - dn_before, c1 * beta * np.sum(jac * dv))
                    if dn - dn_before <= - c1 * beta * dn_before and beta * np.sum(jac * dv) < 0:
                        L_old = L
                        dn_before = dn
                        self.v_output = v_temp
                        break
            elif line_search_method == "BackTrackingD":
                beta = 2
                LineSearch_converge_flag = False

                while True:
                    beta *= 0.5
                    if beta < BT_beta_threshold:
                        LineSearch_converge_flag = True
                        break
                    # Traditional WuYang
                    v_temp = self.v_output + beta * dv
                    L = self.Lagrangian_WuYang(v=v_temp)
                    print(beta, L - L_old, c1 * beta * np.sum(jac * dv))
                    if L - L_old <= c1 * beta * np.sum(jac * dv) and \
                            beta * np.sum(jac * dv) < 0:
                        n = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
                        dn = np.sum(np.abs(n - n_input) * self.molecule.w)
                        print(beta, L - L_old, dn - dn_before, c1 * beta * np.sum(jac * dv))
                        if dn - dn_before <= - c1 * beta * dn_before:
                            L_old = L
                            dn_before = dn
                            self.v_output = v_temp
                            break
            elif line_search_method == "StrongWolfe":
                LineSearch_converge_flag = False

                beta,_,_,L,_,_ = optimizer.line_search(self.Lagrangian_WuYang,
                                                       self.grad_WuYang, self.v_output, dv,
                                                       old_fval=L_old, gfk=jac, maxiter=100,
                                                       c1=c1,
                                                       c2=c2,
                                                       amax=10
                                                       )
                if beta is None:
                    # beta = 0
                    LineSearch_converge_flag = True
                else:
                    self.v_output += beta * dv
                    if svd_rcond == "GL" or svd_rcond == "GL_mod":
                        self.v0_output -= beta * v0
                        self.vbara_output -= beta * v_bara
                        self.vbarb_output -= beta * v_barb
                        # if scf_step % 3 == 0:
                        #     self.v0_output += beta * dv
                        # elif scf_step % 2 == 0:
                        #     self.vbara_output += beta * dv
                        # else:
                        #     self.vbara_output += beta * dv
                    n = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
                    dn = np.sum(np.abs(n - n_input) * self.molecule.w)
                    print(beta, L - L_old, dn - dn_before)
                    dn_before = dn
                    L_old = L

            elif line_search_method == "StrongWolfeD":
                LineSearch_converge_flag = False

                beta,_,_,L,_,_ = optimizer.line_search(self.Lagrangian_WuYang,
                                                       self.grad_WuYang, self.v_output, dv,
                                                       old_fval=L_old, gfk=jac, maxiter=100,
                                                       c1=c1,
                                                       c2=c2,
                                                       extra_condition=density_improvement_test,
                                                       amax=10)
                if beta is None:
                    LineSearch_converge_flag = True
                else:
                    self.v_output += beta * dv
                    if svd_rcond == "GL" or svd_rcond == "GL_mod":
                        self.v0_output -= beta * v0
                        self.vbara_output -= beta * v_bara
                        self.vbarb_output -= beta * v_barb
                    n = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
                    dn = np.sum(np.abs(n - n_input) * self.molecule.w)
                    print(beta, L - L_old, dn - dn_before)
                    dn_before = dn
                    L_old = L
            print(
                F'------Iter: {scf_step} BT: {line_search_method} SVD: {self.svd_index} Reg: {self.regularization_constant} '
                F'Ortho: {self.ortho_basis} SVDmoveon: {LineSearch_converge_flag} ------\n'
                F'beta: {beta} |jac|: {np.linalg.norm(jac)} L: {L_old} d_rho: {dn_before} '
                F'eHOMO: {self.molecule.eig_a.np[self.molecule.nalpha-1], self.molecule.eig_b.np[self.molecule.nbeta-1]}\n')
            print("|vbar|", np.linalg.norm(self.vbara_output+self.vbarb_output))
            if LineSearch_converge_flag and \
                    not (svd_rcond=="segments" or svd_rcond=="segment_cycle"
                         or svd_rcond=="increase" or svd_rcond=="input_segment_cycle"
                         or svd_rcond=="search_segment_cycle"
                         or svd_rcond=="search_cycle_segment"
                         or svd_rcond=="segment_cycle_cutoff"
                    ):
                print("Converge")
                break
            elif dn_before < rho_conv_threshold:
                print("Break because rho difference (cost) is small.")
                break
            elif scf_step == maxiter:
                print("Maximum number of SCF cycles exceeded for vp.")


        # if svd_rcond == "find_optimal_w_bruteforce":
        #     return rcondlist, dnlist, Llist

        print("Evaluation: ", self.L_counter, self.grad_counter, self.hess_counter)
        print("Ts", self.molecule.Da.vector_dot(self.molecule.T)+self.molecule.Db.vector_dot(self.molecule.T))
        print("dTs", np.trace(np.dot(self.input_density_wfn.Da().np+self.input_density_wfn.Db().np-
                                     self.molecule.Da.np-self.molecule.Db.np, self.molecule.T.np)))
        print("eigenA input", self.input_density_wfn.epsilon_a().np[:self.molecule.nalpha])
        print("eigenA mol", self.molecule.eig_a.np[:self.molecule.nalpha])
        print("wfnDiff", self.input_density_wfn.Ca().vector_dot(self.molecule.Ca)
              /np.linalg.norm(self.input_density_wfn.Ca().np)/np.linalg.norm(self.molecule.Ca.np))
        print("Constant potential: ", self.vout_constant)

        self.L_counter = 0
        self.grad_counter = 0
        self.hess_counter = 0

        # if find_vxc_grid:
        #     self.get_vxc()

        return hess, jac

# %% PDE constrained optimization on basis sets.
    def Lagrangian_constrainedoptimization(self, v=None):
        """
        Return Lagrangian for Nafziger and Jensen's constrained optimization.
        :return: L
        """
        self.L_counter += 1

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:

                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        if v is not None:
            nbf = int(v.shape[0]/2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            self.molecule.KS_solver(1000, [Vks_a, Vks_b])
            self.update_vout_constant()

        dDa = self.molecule.Da.np - self.input_density_wfn.Da().np
        dDb = self.molecule.Db.np - self.input_density_wfn.Db().np
        dD = dDa + dDb

        g_uv = self.CO_weighted_cost()

        L = np.sum(dD * g_uv)

        # Regularization
        if self.regularization_constant is not None:
            T = self.vp_basis.T.np
            if v is not None:
                norm = 2 * np.dot(np.dot(v_output_a, T), v_output_a) + 2 * np.dot(np.dot(v_output_b, T), v_output_b)
            else:
                nbf = int(self.v_output.shape[0] / 2)
                norm = 2 * np.dot(np.dot(self.v_output[:nbf], T), self.v_output[:nbf]) + \
                       2 * np.dot(np.dot(self.v_output[nbf:], T), self.v_output[nbf:])
            L += norm * self.regularization_constant
            self.regul_norm = norm
        return L

    def grad_constrainedoptimization(self, v=None):
        """
        To get Jaccobi vector, which is jac_j = sum_i p_i*psi_i*phi_j.
        p_i = sum_j b_ij*phi_j
        MO: psi_i = sum_j C_ji*phi_j
        AO: phi
        --
        A: frag, a: spin, i: MO
        """
        self.grad_counter += 1

        if v is not None:
            nbf = int(v.shape[0] / 2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)

            self.molecule.KS_solver(100, [Vks_a, Vks_b])
            self.update_vout_constant()

        # gradient on grad
        jac_real_up = np.zeros((self.molecule.nbf, self.molecule.nbf))
        jac_real_down = np.zeros((self.molecule.nbf, self.molecule.nbf))

        # Pre-calculate \int (n - nf)*phi_u*phi_v
        g_uv = - self.CO_weighted_cost()
        # Fragments
        A = self.molecule
        # spin up
        for i in range(A.nalpha):
            u = 2 * np.einsum("u,v,uv->", A.Cocca.np[:,i], A.Cocca.np[:,i], g_uv)
            LHS = A.Fa.np - A.S.np*A.eig_a.np[i]
            RHS = 4 * np.einsum("uv,v->u", g_uv, A.Cocca.np[:,i]) - 2 * u * np.dot(A.S.np, A.Cocca.np[:,i])
            p = np.linalg.solve(LHS, RHS)
            # s = np.linalg.svd(LHS)[1]
            # if self.svd_index is None:
            #     s = np.linalg.svd(LHS)[1]
            #     print(repr(s))
            #
            #     f, ax = plt.subplots(1,1,dpi=200)
            #     ax.scatter(range(s.shape[0]), np.log10(s), s=3)
            #     ax.set_title(str(self.ortho_basis))
            #     f.show()
            #     plt.close()
            #
            #     self.svd_index = int(input("Enter svd cut index (0-based indexing): "))
            #
            # p = np.dot(np.linalg.pinv(LHS, rcond=s[self.svd_index]*1.01/s[0]), RHS)

            # Gram–Schmidt
            p = p - np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])) * A.Cocca.np[:,i]
            # assert np.allclose([np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Cocca.np[:,i])], 0, atol=1e-4), \
            #     [np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Cocca.np[:,i])]
            jac_real_up += np.dot(p[:, None], A.Cocca.np[:,i:i+1].T)
            del p,u,LHS,RHS

        # spin down
        for i in range(A.nbeta):
            u = 2 * np.einsum("u,v,uv->", A.Coccb.np[:,i], A.Coccb.np[:,i], g_uv)
            LHS = A.Fb.np - A.S.np*A.eig_b.np[i]
            RHS = 4 * np.einsum("uv,v->u", g_uv, A.Coccb.np[:,i]) - 2 * u * np.dot(A.S.np, A.Coccb.np[:,i])
            p = np.linalg.solve(LHS, RHS)
            # s = np.linalg.svd(LHS)[1]
            # p = np.dot(np.linalg.pinv(LHS, rcond=s[self.svd_index]/s[0]*1.01), RHS)
            #
            # Gram–Schmidt
            p = p - np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i]))*A.Coccb.np[:,i]
            # assert np.allclose([np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Coccb.np[:,i])], 0, atol=1e-4), \
            #     [np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Coccb.np[:,i])]
            jac_real_down += np.dot(p[:, None], A.Coccb.np[:,i:i+1].T)
            del p,u,LHS,RHS

        # jac = int jac_real*phi_w
        jac_up = np.einsum("uv,uvw->w", jac_real_up, self.three_overlap)
        jac_down = np.einsum("uv,uvw->w", jac_real_down, self.three_overlap)

        # Regularization
        if self.regularization_constant is not None:
            T = self.vp_basis.T.np
            if v is not None:
                jac_up += 4*self.regularization_constant*np.dot(T, v_output_a)
                jac_down += 4*self.regularization_constant*np.dot(T, v_output_b)
            else:
                nbf = int(self.v_output.shape[0] / 2)
                jac_up += 4*self.regularization_constant*np.dot(T, self.v_output[:nbf])
                jac_down += 4*self.regularization_constant*np.dot(T, self.v_output[nbf:])
        return np.concatenate((jac_up, jac_down))

    def hess_constrainedoptimization(self, v=None):
        # DOES NOT WORK NOW!
        self.hess_counter += 1

        if v is not None:
            nbf = int(v.shape[0] / 2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)

            self.molecule.KS_solver(100, [Vks_a, Vks_b])
            self.update_vout_constant()

        hess = np.zeros((self.vp_basis.nbf*2, self.vp_basis.nbf*2))

        # Pre-calculate \int (n - nf)*phi_u*phi_v
        g_uv = - self.CO_weighted_cost()
        # Fragments
        A = self.molecule
        # spin up
        for i in range(A.nalpha):
            pi_psi_up = np.zeros((self.vp_basis.nbf, self.vp_basis.nbf))
            psii_psi_up = np.zeros((self.vp_basis.nbf, self.vp_basis.nbf))

            u = 2 * np.einsum("u,v,uv->", A.Cocca.np[:,i], A.Cocca.np[:,i], g_uv)
            LHS = A.Fa.np - A.S.np*A.eig_a.np[i]
            RHS = 4 * np.einsum("uv,v->u", g_uv, A.Cocca.np[:,i]) - 2 * u * np.dot(A.S.np, A.Cocca.np[:,i])
            p = np.linalg.solve(LHS, RHS)
            # s = np.linalg.svd(LHS)[1]
            # if self.svd_index is None:
            #     s = np.linalg.svd(LHS)[1]
            #     print(repr(s))
            #
            #     f, ax = plt.subplots(1,1,dpi=200)
            #     ax.scatter(range(s.shape[0]), np.log10(s), s=3)
            #     ax.set_title(str(self.ortho_basis))
            #     f.show()
            #     plt.close()
            #
            #     self.svd_index = int(input("Enter svd cut index (0-based indexing): "))
            #
            # p = np.dot(np.linalg.pinv(LHS, rcond=s[self.svd_index]*1.01/s[0]), RHS)

            # Gram–Schmidt
            p = p - np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])) * A.Cocca.np[:,i]
            assert np.allclose([np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Cocca.np[:,i])], 0, atol=1e-4), \
                [np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Cocca.np[:,i])]

            for j in range(A.nbf):
                if i == j:
                    continue
                # p_i * psi_j
                pi_psi_up = np.dot(p[:, None], A.Ca.np[:,j:j+1].T)
                # psi_i * psi_j / (Ei-Ej)
                psii_psi_up = np.dot(A.Cocca.np[:,i:i+1], A.Ca.np[:,j:j+1].T) / (A.eig_a.np[i] - A.eig_a.np[j])
                assert psii_psi_up.shape == pi_psi_up.shape

                hess[0:self.vp_basis.nbf, 0:self.vp_basis.nbf] += np.einsum("mn,uv,mni,uvj->ij", pi_psi_up, psii_psi_up,
                                                                            self.three_overlap, self.three_overlap,
                                                                            optimize=True)

        # spin down
        for i in range(A.nbeta):
            pi_psi_dw = np.zeros((self.vp_basis.nbf, self.vp_basis.nbf))
            psii_psi_dw = np.zeros((self.vp_basis.nbf, self.vp_basis.nbf))

            u = 2 * np.einsum("u,v,uv->", A.Coccb.np[:,i], A.Coccb.np[:,i], g_uv)
            LHS = A.Fb.np - A.S.np*A.eig_b.np[i]
            RHS = 4 * np.einsum("uv,v->u", g_uv, A.Coccb.np[:,i]) - 2 * u * np.dot(A.S.np, A.Coccb.np[:,i])
            p = np.linalg.solve(LHS, RHS)
            # s = np.linalg.svd(LHS)[1]
            # p = np.dot(np.linalg.pinv(LHS, rcond=s[self.svd_index]/s[0]*1.01), RHS)

            # Gram–Schmidt
            p = p - np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i]))*A.Coccb.np[:,i]
            assert np.allclose([np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Coccb.np[:,i])], 0, atol=1e-4), \
                [np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Coccb.np[:,i])]
            for j in range(A.nbf):
                if i == j:
                    continue
                # p_i * psi_j
                pi_psi_dw = np.dot(p[:, None], A.Cb.np[:, j:j + 1].T)
                # psi_i * psi_j / (Ei-Ej)
                psii_psi_dw = np.dot(A.Coccb.np[:, i:i + 1], A.Cb.np[:, j:j + 1].T) / (A.eig_b.np[i] - A.eig_b.np[j])

                hess[self.vp_basis.nbf:, self.vp_basis.nbf:] += np.einsum("mn,uv,mni,uvj->ij", pi_psi_dw, psii_psi_dw,
                                                                          self.three_overlap, self.three_overlap,
                                                                          optimize=True)
        return hess

    def CO_weighted_cost(self):
        """
        To get the function g_uv = \int w*(n-n_in)*phi_u*phi_v.
        So that the cost function is dD_uv*g_uv.
        The first term in g funtion is -2*C_v*g_uv.
        w = n_mol**-self.CO_weight_exponent
        :return:
        """
        if self.four_overlap is None:
            self.four_overlap = pdft.fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                 self.molecule.basis, self.molecule.mints)[0]

        D_mol = self.input_density_wfn.Da().np + self.input_density_wfn.Db().np

        dDa = self.molecule.Da.np - self.input_density_wfn.Da().np
        dDb = self.molecule.Db.np - self.input_density_wfn.Db().np
        dD = dDa + dDb

        g_uv = np.zeros_like(self.molecule.H.np)
        if self.CO_weight_exponent is None:
            return np.einsum("mn,mnuv->uv", dD, self.four_overlap, optimize=True)
        else:
            print("WEIGHT NOT BE 1 IS HIGHLY UNRECOMMENDED SINCE THE ASYMPTOTIC BEHAVIOR "
                  "IS ALREADY FIXED BY THE BASIS SET AND GUIDE POTENTIAL!")
            vpot = self.molecule.Vpot
            points_func = vpot.properties()[0]
            f_grid = np.array([])
            # Loop over the blocks
            for b in range(vpot.nblocks()):
                # Obtain block information
                block = vpot.get_block(b)
                points_func.compute_points(block)
                w = block.w()
                npoints = block.npoints()
                lpos = np.array(block.functions_local_to_global())

                # Compute phi!
                phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

                # Build a local slice of D
                ldD = dD[(lpos[:, None], lpos)]
                lD_mol = D_mol[(lpos[:, None], lpos)]

                # Copmute dn and n_mol
                n_mol = np.einsum('pm,mn,pn->p', phi, lD_mol, phi)

                g_uv[(lpos[:, None], lpos)] += np.einsum('pm,pn,pu,pv,p,mn,p->uv', phi, phi, phi, phi, (1/n_mol)**self.CO_weight_exponent,
                                  ldD, w, optimize=True)
            return g_uv

    def CO_p(self, v=None):
        if v is not None:
            nbf = int(v.shape[0] / 2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)

            self.molecule.KS_solver(100, [Vks_a, Vks_b])
            self.update_vout_constant()

        p_up = np.zeros((self.molecule.nbf, self.molecule.nalpha))
        p_dw = np.zeros((self.molecule.nbf, self.molecule.nbeta))

        # Pre-calculate \int (n - nf)*phi_u*phi_v
        g_uv = - self.CO_weighted_cost()
        # Fragments
        A = self.molecule
        # spin up
        for i in range(A.nalpha):
            u = 2 * np.einsum("u,v,uv->", A.Cocca.np[:,i], A.Cocca.np[:,i], g_uv)
            LHS = A.Fa.np - A.S.np*A.eig_a.np[i]
            RHS = 4 * np.einsum("uv,v->u", g_uv, A.Cocca.np[:,i]) - 2 * u * np.dot(A.S.np, A.Cocca.np[:,i])
            p = np.linalg.solve(LHS, RHS)
            # s = np.linalg.svd(LHS)[1]
            # if self.svd_index is None:
            #     s = np.linalg.svd(LHS)[1]
            #     print(repr(s))
            #
            #     f, ax = plt.subplots(1,1,dpi=200)
            #     ax.scatter(range(s.shape[0]), np.log10(s), s=3)
            #     ax.set_title(str(self.ortho_basis))
            #     f.show()
            #     plt.close()
            #
            #     self.svd_index = int(input("Enter svd cut index (0-based indexing): "))
            #
            # p = np.dot(np.linalg.pinv(LHS, rcond=s[self.svd_index]*1.01/s[0]), RHS)

            # Gram–Schmidt
            p = p - np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])) * A.Cocca.np[:,i]
            assert np.allclose([np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Cocca.np[:,i])], 0, atol=1e-4), \
                [np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Cocca.np[:,i])]

            p_up[:, i] = p

        # spin down
        for i in range(A.nbeta):
            u = 2 * np.einsum("u,v,uv->", A.Coccb.np[:,i], A.Coccb.np[:,i], g_uv)
            LHS = A.Fb.np - A.S.np*A.eig_b.np[i]
            RHS = 4 * np.einsum("uv,v->u", g_uv, A.Coccb.np[:,i]) - 2 * u * np.dot(A.S.np, A.Coccb.np[:,i])
            p = np.linalg.solve(LHS, RHS)
            # s = np.linalg.svd(LHS)[1]
            # p = np.dot(np.linalg.pinv(LHS, rcond=s[self.svd_index]/s[0]*1.01), RHS)

            # Gram–Schmidt
            p = p - np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i]))*A.Coccb.np[:,i]
            assert np.allclose([np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Coccb.np[:,i])], 0, atol=1e-4), \
                [np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Coccb.np[:,i])]

            p_dw[:, i] = p
        return p_up, p_dw

    def check_gradient_constrainedoptimization(self, dv=None):
        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:

                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        nbf = int(self.v_output.shape[0] / 2)
        v_output_a = self.v_output[:nbf]
        v_output_b = self.v_output[nbf:]

        Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        self.molecule.KS_solver(100, [Vks_a, Vks_b])
        self.update_vout_constant()

        L = self.Lagrangian_constrainedoptimization()
        grad = self.grad_constrainedoptimization()

        if dv is None:
            dv = 1e-7*np.ones_like(self.v_output)

        grad_app = np.zeros_like(dv)

        for i in range(dv.shape[0]):
            dvi = np.zeros_like(dv)
            dvi[i] = dv[i]

            L_new = self.Lagrangian_constrainedoptimization(dvi+self.v_output)

            grad_app[i] = (L_new-L) / dvi[i]

            # print(L_new, L, i + 1, "out of ", dv.shape[0])

        print(np.sum(grad*grad_app)/np.linalg.norm(grad)/np.linalg.norm(grad_app))
        print(np.linalg.norm(grad_app-grad)/np.linalg.norm(grad))

        return grad, grad_app

    def check_hess_constrainedoptimization(self, dv=None):
        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:

                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        nbf = int(self.v_output.shape[0] / 2)
        v_output_a = self.v_output[:nbf]
        v_output_b = self.v_output[nbf:]

        Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        self.molecule.KS_solver(100, [Vks_a, Vks_b])
        self.update_vout_constant()

        hess = self.hess_constrainedoptimization()
        grad = self.grad_constrainedoptimization()

        if dv is None:
            dv = 1e-7*np.ones_like(self.v_output)

        hess_app = np.zeros_like(hess)

        for i in range(grad.shape[0]):
            dvi = np.zeros_like(dv)
            dvi[i] = dv[i]

            grad_new = self.grad_constrainedoptimization(dvi)

            hess_app[i,:] = (grad_new - grad)/dv[i]

        hess_app = 0.5 * (hess_app + hess_app.T)
        print("SYMMETRY", np.linalg.norm(hess - hess.T))
        print("DIRECTION", np.trace(hess_app.dot(hess.T))/np.linalg.norm(hess_app)/np.linalg.norm(hess))
        print("SIMILARITY", np.linalg.norm(hess - hess_app))
        return hess, hess_app

    def find_vxc_scipy_constrainedoptimization(self, maxiter=1400, opt_method="BFGS",
                                               continue_opt=False, find_vxc_grid=False, opt=None):

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:

                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        if not continue_opt:
            print("Zero the old result for a new calculation..")
            self.v_output = np.zeros_like(self.v_output)

            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.KS_solver(100, [Vks_a, Vks_b])
            self.update_vout_constant()

        print("<<<<<<<<<<<<<<<<<<<<<<Constrained Optimization vxc Inversion<<<<<<<<<<<<<<<<<<<")

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("|n| before", np.sum(np.abs(dn)*self.molecule.w))
        if opt is None:
            opt = {
                "disp": False,
                'maxls': 2000,
            }
        opt["maxiter"] = maxiter

        result_x = optimizer.minimize(self.Lagrangian_constrainedoptimization,
                                      self.v_output,
                                      jac=self.grad_constrainedoptimization,
                                      method=opt_method,
                                      options=opt)

        nbf = int(result_x.x.shape[0] / 2)
        v_output_a = result_x.x[:nbf]
        v_output_b = result_x.x[nbf:]

        Vks_a = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)

        self.molecule.KS_solver(100, [Vks_a, Vks_b])
        self.update_vout_constant()

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("Evaluation", result_x.nfev)
        print("|jac|", np.linalg.norm(result_x.jac), "|n|", np.sum(np.abs(dn)*self.molecule.w), "L after", result_x.fun)
        print("Ts", self.molecule.Da.vector_dot(self.molecule.T)+self.molecule.Db.vector_dot(self.molecule.T))
        print("dTs", np.trace(np.dot(self.input_density_wfn.Da().np+self.input_density_wfn.Db().np-
                                     self.molecule.Da.np-self.molecule.Db.np, self.molecule.T.np)))
        print("eigenA")
        print(self.input_density_wfn.epsilon_a().np[:self.molecule.nalpha])
        print(self.molecule.eig_a.np[:self.molecule.nalpha])
        print("wfnDiff", self.input_density_wfn.Ca().vector_dot(self.molecule.Ca)
              /np.linalg.norm(self.input_density_wfn.Ca().np)/np.linalg.norm(self.molecule.Ca.np))
        print("Constant potential: ", self.vout_constant)
        # Update info
        self.v_output = result_x.x

        if find_vxc_grid:
            self.get_vxc()
        return result_x

    def my_L_curve_regularization4CO(self, rgl_bs=np.e, rgl_epn=15, starting_epn=1,
                                  searching_method="close_to_platform",
                                  close_to_platform_rtol=0.001,
                                  scipy_opt_method="L-BFGS-B", print_flag=True):

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)
        v_zero_initial = np.zeros_like(self.v_output)

        # Loop
        L_list = []
        error_list = []
        P_list = []
        rgl_order = starting_epn + np.array(range(rgl_epn))
        rgl_list =  rgl_bs**-(rgl_order+2)
        rgl_list = np.append(rgl_list, 0)
        n_input = self.molecule.to_grid(self.input_density_wfn.Da().np + self.input_density_wfn.Db().np)

        print("Start L-curve search for regularization constant lambda. This might take a while..")
        for regularization_constant in rgl_list:
            self.regularization_constant = regularization_constant
            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.KS_solver(100, [Vks_a, Vks_b])
            self.update_vout_constant()

            opt = {
                "disp": False,
                "maxiter": 10000,
            }

            v_result = optimizer.minimize(self.Lagrangian_constrainedoptimization, v_zero_initial,
                                          jac=self.grad_constrainedoptimization,
                                          method=scipy_opt_method,
                                          options=opt)

            v = v_result.x
            nbf = int(v.shape[0]/2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]
            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            self.molecule.KS_solver(1000, [Vks_a, Vks_b])
            self.update_vout_constant()

            n_result = self.molecule.to_grid(self.molecule.Da.np + self.molecule.Db.np)
            P = np.linalg.norm(v_output_a) * np.sum(np.abs(n_result - n_input) * self.molecule.w)
            P_list.append(P)
            L_list.append(v_result.fun)
            error_list.append(np.sum(self.molecule.w*(n_result - n_input)**2))
            print(regularization_constant, "P", P, "T", error_list[-1])

        # L-curve
        error_list = np.array(error_list)

        f, ax = plt.subplots(1, 1, dpi=200)
        ax.scatter(range(error_list.shape[0]), error_list)
        f.show()
        if searching_method == "std_error_min":
            start_idx = int(input("Enter start index for the left line of L-curve: "))

            r_list = []
            std_list = []
            for i in range(start_idx + 3, error_list.shape[0] - 2):
                left = error_list[start_idx:i]
                right = error_list[i + 1:]
                left_x = range(start_idx, i)
                right_x = range(i + 1, error_list.shape[0])
                slopl, intl, rl, _, stdl = stats.linregress(left_x, left)
                slopr, intr, rr, _, stdr = stats.linregress(right_x, right)
                if print_flag:
                    print(i, stdl + stdr, rl + rr, "Right:", slopr, intr, "Left:", slopl, intl)
                r_list.append(rl + rr)
                std_list.append(stdl + stdr)

            # The final index
            i = np.argmin(std_list) + start_idx + 3
            self.regularization_constant = rgl_list[i]
            print("Regularization constant lambda from L-curve is ", self.regularization_constant)

            if print_flag:
                left = error_list[start_idx:i]
                right = error_list[i + 1:]
                left_x = range(start_idx, i)
                right_x = range(i + 1, error_list.shape[0])
                slopl, intl, rl, _, stdl = stats.linregress(left_x, left)
                slopr, intr, rr, _, stdr = stats.linregress(right_x, right)
                x = np.array(ax.get_xlim())
                yl = intl + slopl * x
                yr = intr + slopr * x
                ax.plot(x, yl, '--')
                ax.plot(x, yr, '--')
                ax.set_ylim(np.min(error_list)*0.99, np.max(error_list)*1.01)
                f.show()

        elif searching_method == "close_to_platform":
            for i in range(len(error_list)):
                if np.abs(np.log(error_list[i]) - np.log(error_list[-1])/np.log(error_list[-1])) <= close_to_platform_rtol:
                    self.regularization_constant = rgl_list[i]
                    print("Regularization constant lambda from L-curve is ", self.regularization_constant)
                    break
            if print_flag:
                ax.axhline(y=error_list[-1], ls="--", lw=0.7, color='r')
                ax.scatter(i, error_list[i], marker="+")
                f.show()
        return rgl_list, L_list, error_list, P_list

# %% Get vxc on the grid. DOES NOT WORK NOW.
    def Lagrangian_WuYang_grid(self, v=None):
        """
        L = - <T> - \int (vks_a*(n_a-n_a_input)+vks_b*(n_b-n_b_input))
        :return: L
        """
        self.L_counter += 1

        if v is not None:
            npt = int(v.shape[0]/2)
            v_output_a = v[:npt]
            v_output_b = v[npt:]

            Vks_a = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            self.molecule.KS_solver(1000, [Vks_a, Vks_b])
            self.update_vout_constant()

        L = - self.molecule.T.vector_dot(self.molecule.Da) - self.molecule.T.vector_dot(self.molecule.Db)
        L += - self.molecule.vks_a.vector_dot(self.molecule.Da) - self.molecule.vks_b.vector_dot(self.molecule.Db)
        L += self.molecule.vks_a.vector_dot(self.input_density_wfn.Da()) + self.molecule.vks_b.vector_dot(self.input_density_wfn.Db())
        return L

    def grad_WuYang_grid(self, v=None):
        """
        grad_a = dL/dvxc_a = - (n_a-n_a_input)
        grad_b = dL/dvxc_b = - (n_b-n_b_input)
        :return:
        """
        self.grad_counter += 1

        if v is not None:
            npt = int(v.shape[0]/2)
            v_output_a = v[:npt]
            v_output_b = v[npt:]

            Vks_a = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            self.molecule.KS_solver(1000, [Vks_a, Vks_b])
            self.update_vout_constant()

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np

        grad_a = self.molecule.to_grid(dDa)
        grad_b = self.molecule.to_grid(dDb)
        grad = np.concatenate((grad_a, grad_b))

        return grad

    def check_gradient_WuYang_grid(self, dv=None):
        npt = int(self.v_output_grid.shape[0] / 2)
        v_output_a = self.v_output_grid[:npt]
        v_output_b = self.v_output_grid[npt:]

        Vks_a = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        self.molecule.KS_solver(100, [Vks_a, Vks_b])
        self.update_vout_constant()

        L = self.Lagrangian_WuYang_grid()
        grad = self.grad_WuYang_grid()

        if dv is None:
            dv = 1e-7*np.ones_like(self.v_output_grid)

        grad_app = np.zeros_like(dv)

        print("Start testing. This will time a while...")
        dvsize = dv.shape[0]
        dvsizepercentile = int(dvsize/10)
        for i in range(dvsize):
            if (i%dvsizepercentile) == 0:
                print("%i %% done." %(i/dvsizepercentile*10))
            dvi = np.zeros_like(dv)
            dvi[i] = dv[i]

            L_new = self.Lagrangian_WuYang_grid(dvi+self.v_output_grid)

            grad_app[i] = (L_new-L) / dvi[i]
        print(np.sum(grad*grad_app)/np.linalg.norm(grad)/np.linalg.norm(grad_app))
        print(np.linalg.norm(grad_app-grad)/np.linalg.norm(grad))

        return grad, grad_app

    def WuYang_LandGradient_grid_wrapper(self, v, g):
        """
        A wrapper to combine L and grad on grid for pylbfgs optimizer
        :param v: x
        :param g: gradient
        :return:
        """
        L = self.Lagrangian_WuYang_grid(v)
        g[:] = self.grad_WuYang_grid(v)
        return L

    def find_vxc_pylbfgs_WuYang_grid(self, maxiter=14000, line_search='strongwolfe', ftol=1e-4,
                                   continue_opt=False,
                                   max_linesearch=100,
                                   find_vxc_grid=True):
        if not continue_opt:
            print("Zero the old result for a new calculation..")
            self.v_output_grid = np.zeros_like(self.v_output_grid)
            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.KS_solver(100, [Vks_a, Vks_b])
            self.update_vout_constant()

        print("<<<<<<<<<<<<<<<<<<<<<<WuYang vxc Inversion<<<<<<<<<<<<<<<<<<<")

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa + dDb)
        print("|n| before", np.sum(np.abs(dn) * self.molecule.w))

        v = fmin_lbfgs(self.WuYang_LandGradient_grid_wrapper, self.v_output_grid,
                               max_step=maxiter,
                               line_search=line_search,
                               ftol=ftol,
                               max_linesearch=max_linesearch)

        nbf = int(v.shape[0] / 2)
        v_output_a = v[:nbf]
        v_output_b = v[nbf:]

        Vks_a = psi4.core.Matrix.from_array(
            self.molecule.grid_to_fock(v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(
            self.molecule.grid_to_fock(v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        self.molecule.KS_solver(1000, [Vks_a, Vks_b])
        self.update_vout_constant()

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa + dDb)
        print("|jac|", np.linalg.norm(self.grad_WuYang_grid(v)), "|n|", np.sum(np.abs(dn) * self.molecule.w), "L after",
              self.Lagrangian_WuYang_grid(v))
        print("Ts", self.molecule.Da.vector_dot(self.molecule.T) + self.molecule.Db.vector_dot(self.molecule.T))
        print("dTs", np.trace(np.dot(self.input_density_wfn.Da().np + self.input_density_wfn.Db().np -
                                     self.molecule.Da.np - self.molecule.Db.np, self.molecule.T.np)))
        print("eigenA")
        print(self.input_density_wfn.epsilon_a().np[:self.molecule.nalpha])
        print(self.molecule.eig_a.np[:self.molecule.nalpha])
        print("wfnDiff", self.input_density_wfn.Ca().vector_dot(self.molecule.Ca)
              / np.linalg.norm(self.input_density_wfn.Ca().np) / np.linalg.norm(self.molecule.Ca.np))
        print("Constant potential: ", self.vout_constant)

        # Update info
        self.v_output_grid = v

        if find_vxc_grid:
            self.get_vxc_grid()
        return

    def find_vxc_scipy_WuYang_grid(self, maxiter=14000, opt_method="L-BFGS-B", opt=None,
                              continue_opt=False, find_vxc_grid=True):
        if not continue_opt:
            print("Zero the old result for a new calculation..")
            self.v_output_grid = np.zeros_like(self.v_output_grid)
            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.KS_solver(100, [Vks_a, Vks_b])
            self.update_vout_constant()

        print("<<<<<<<<<<<<<<<<<<<<<<WuYang vxc Inversion<<<<<<<<<<<<<<<<<<<")

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("|n| before", np.sum(np.abs(dn)*self.molecule.w))
        if opt is None:
            opt = {
                "disp": True,
                "maxiter": maxiter,
                'maxls': 2000,
                # "eps": 1e-7
                # "norm": 2,
                # "gtol": 1e-7
            }

        result_x = optimizer.minimize(self.Lagrangian_WuYang_grid, self.v_output_grid,
                                      jac=self.grad_WuYang_grid,
                                      method=opt_method,
                                      options=opt)
        nbf = int(result_x.x.shape[0] / 2)
        v_output_a = result_x.x[:nbf]
        v_output_b = result_x.x[nbf:]

        Vks_a = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        self.molecule.KS_solver(1000, [Vks_a, Vks_b])
        self.update_vout_constant()

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("|jac|", np.linalg.norm(result_x.jac), "|n|", np.sum(np.abs(dn)*self.molecule.w), "L after", result_x.fun)
        print("Ts", self.molecule.Da.vector_dot(self.molecule.T)+self.molecule.Db.vector_dot(self.molecule.T))
        print("dTs", np.trace(np.dot(self.input_density_wfn.Da().np+self.input_density_wfn.Db().np-
                                     self.molecule.Da.np-self.molecule.Db.np, self.molecule.T.np)))
        print("eigenA")
        print(self.input_density_wfn.epsilon_a().np[:self.molecule.nalpha])
        print(self.molecule.eig_a.np[:self.molecule.nalpha])
        print("wfnDiff", self.input_density_wfn.Ca().vector_dot(self.molecule.Ca)
              /np.linalg.norm(self.input_density_wfn.Ca().np)/np.linalg.norm(self.molecule.Ca.np))
        print("Constant potential: ", self.vout_constant)

        # Update info
        self.v_output = result_x.x

        if find_vxc_grid:
            self.get_vxc_grid()
        return
#%% Get vxc on the density basis set. DOES NOT WORK NOW.
    def Lagrangian_WuYang_grid1(self, v=None):
        """

        :return: L
        """
        if self.four_overlap is None:
            self.four_overlap, _, _, _ = pdft.fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                          self.molecule.basis, self.molecule.mints)

        assert not self.ortho_basis, "Does not support Orthogonal Basis Set"

        self.L_counter += 1

        if v is not None:
            npt = int(v.shape[0]/2)
            v_output_a = v[:npt]
            v_output_b = v[npt:]

            v_output_a.shape = (self.molecule.nbf, self.molecule.nbf)
            v_output_b.shape = (self.molecule.nbf, self.molecule.nbf)

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijkl,kl->ij", self.four_overlap, v_output_a) +
                                                self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijkl,kl->ij", self.four_overlap, v_output_b) +
                                                self.vout_constant * self.molecule.S.np + self.v0_Fock)
            self.molecule.KS_solver(1000, [Vks_a, Vks_b])
            self.update_vout_constant()

        L = - self.molecule.T.vector_dot(self.molecule.Da) - self.molecule.T.vector_dot(self.molecule.Db)
        # L += - self.molecule.vks_a.vector_dot(self.molecule.Da) - self.molecule.vks_b.vector_dot(self.molecule.Db)
        # L += self.molecule.vks_a.vector_dot(self.input_density_wfn.Da()) + self.molecule.vks_b.vector_dot(self.input_density_wfn.Db())
        return L

    def grad_WuYang_grid1(self, v=None):
        """
        grad_a = dL/dvxc_a = - (n_a-n_a_input)
        grad_b = dL/dvxc_b = - (n_b-n_b_input)
        :return:
        """
        if self.four_overlap is None:
            self.four_overlap, _, _, _ = pdft.fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                          self.molecule.basis, self.molecule.mints)
        self.grad_counter += 1

        if v is not None:
            npt = int(v.shape[0]/2)
            v_output_a = v[:npt]
            v_output_b = v[npt:]

            v_output_a.shape = (self.molecule.nbf, self.molecule.nbf)
            v_output_b.shape = (self.molecule.nbf, self.molecule.nbf)

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijkl,kl->ij", self.four_overlap, v_output_a) +
                                                self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijkl,kl->ij", self.four_overlap, v_output_b) +
                                                self.vout_constant * self.molecule.S.np + self.v0_Fock)
            self.molecule.KS_solver(1000, [Vks_a, Vks_b])
            self.update_vout_constant()

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np

        grad_a = dDa
        grad_b = dDb

        # grad_a = np.einsum("ijkl,kl->ij", self.four_overlap, dDa)
        # grad_b = np.einsum("ijkl,kl->ij", self.four_overlap, dDb)

        grad_a.shape = self.molecule.nbf**2
        grad_b.shape = self.molecule.nbf**2

        grad = np.concatenate((grad_a, grad_b))

        return grad

    def check_gradient_WuYang_grid1(self, dv=None):
        if self.four_overlap is None:
            self.four_overlap, _, _, _ = pdft.fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                          self.molecule.basis, self.molecule.mints)

        npt = int(self.v_output_grid1.shape[0] / 2)
        v_output_a = self.v_output_grid1[:npt]
        v_output_b = self.v_output_grid1[npt:]

        Vks_a = psi4.core.Matrix.from_array(np.einsum("ijkl,kl->ij", self.four_overlap,
                                                      np.reshape(v_output_a, (self.molecule.nbf, self.molecule.nbf)))
                                            + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(np.einsum("ijkl,kl->ij", self.four_overlap,
                                                      np.reshape(v_output_b, (self.molecule.nbf, self.molecule.nbf)))
                                            + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        self.molecule.KS_solver(1000, [Vks_a, Vks_b])
        self.update_vout_constant()

        L = self.Lagrangian_WuYang_grid1()
        grad = self.grad_WuYang_grid1()

        if dv is None:
            dv = 1e-7 * np.ones_like(self.v_output_grid1)

        grad_app = np.zeros_like(dv)

        print("Start testing. This will time a while...")
        dvsize = dv.shape[0]
        dvsizepercentile = int(dvsize/10)
        for i in range(dvsize):
            if (i%dvsizepercentile) == 0:
                print("%i %% done." %(i/dvsizepercentile*10))
            dvi = np.zeros_like(dv)
            dvi[i] = dv[i]

            L_new = self.Lagrangian_WuYang_grid1(dvi+self.v_output_grid1)

            # print(L_new-L)
            grad_app[i] = (L_new-L) / dvi[i]
        print(np.sum(grad*grad_app)/np.linalg.norm(grad)/np.linalg.norm(grad_app))
        print(np.linalg.norm(grad_app-grad)/np.linalg.norm(grad))
        return grad, grad_app

#%% Section for finding vext_approximate on grid in order to avoid the singularity in real vext. DOES NOT WORK NOW.
    def find_vext_scipy_WuYang_grid(self, maxiter=14000, opt_method="L-BFGS-B", opt=None,
                              continue_opt=False, find_vxc_grid=True):

        if self.v0 != "HartreeLDAappext":
            print("Changing v0 to Hartree + LDA. Only works for LDA")
            self.get_HartreeLDA_v0()

        if self.four_overlap is None:
            self.four_overlap, _, _, _ = pdft.fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                          self.molecule.basis, self.molecule.mints)

        if not continue_opt:
            print("Zero the old result for a new calculation..")
            if self.vext_app is None:
                self.vext_app = np.zeros(self.molecule.nbf**2)
            self.vext_app = np.zeros_like(self.vext_app)
            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.KS_solver(100, [Vks_a, Vks_b], add_vext=False)
            self.update_vout_constant()

        print("<<<<<<<<<<<<<<<<<<<<<<WuYang vxc Inversion<<<<<<<<<<<<<<<<<<<")

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("|n| before", np.sum(np.abs(dn)*self.molecule.w))
        if opt is None:
            opt = {
                "disp": True,
                "maxiter": maxiter,
                'maxls': 2000,
                # "eps": 1e-7
                # "norm": 2,
                # "gtol": 1e-7
            }

        result_x = optimizer.minimize(self.Lagrangian_vext_WuYang_grid, self.vext_app,
                                      jac=self.grad_vext_WuYang_grid,
                                      method=opt_method,
                                      options=opt)
        nbf = int(result_x.x.shape[0] / 2)
        v_output_a = result_x.x[:nbf]
        v_output_b = result_x.x[nbf:]

        Vks_a = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        self.molecule.KS_solver(1000, [Vks_a, Vks_b], add_vext=False)
        self.update_vout_constant()

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("|jac|", np.linalg.norm(result_x.jac), "|n|", np.sum(np.abs(dn)*self.molecule.w), "L after", result_x.fun)
        print("Ts", self.molecule.Da.vector_dot(self.molecule.T)+self.molecule.Db.vector_dot(self.molecule.T))
        print("dTs", np.trace(np.dot(self.input_density_wfn.Da().np+self.input_density_wfn.Db().np-
                                     self.molecule.Da.np-self.molecule.Db.np, self.molecule.T.np)))
        print("eigenA")
        print(self.input_density_wfn.epsilon_a().np[:self.molecule.nalpha])
        print(self.molecule.eig_a.np[:self.molecule.nalpha])
        print("wfnDiff", self.input_density_wfn.Ca().vector_dot(self.molecule.Ca)
              /np.linalg.norm(self.input_density_wfn.Ca().np)/np.linalg.norm(self.molecule.Ca.np))
        print("Constant potential: ", self.vout_constant)

        # Update info
        self.v_output = result_x.x

        if find_vxc_grid:
            self.get_vxc_grid()
        return

    def Lagrangian_vext_WuYang_grid(self, v=None):
        """

        :return: L
        """
        if self.four_overlap is None:
            self.four_overlap, _, _, _ = pdft.fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                          self.molecule.basis, self.molecule.mints)

        assert not self.ortho_basis, "Does not support Orthogonal Basis Set"

        self.L_counter += 1

        if v is not None:
            reshaped_vext_app = np.reshape(v, (self.molecule.nbf, self.molecule.nbf))

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijkl,kl->ij", self.four_overlap, reshaped_vext_app) +
                                                self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijkl,kl->ij", self.four_overlap, reshaped_vext_app) +
                                                self.vout_constant * self.molecule.S.np + self.v0_Fock)
            self.molecule.KS_solver(1000, [Vks_a, Vks_b], add_vext=False)
            self.update_vout_constant()

        L = - self.molecule.T.vector_dot(self.molecule.Da) - self.molecule.T.vector_dot(self.molecule.Db)
        # L += - self.molecule.vks_a.vector_dot(self.molecule.Da) - self.molecule.vks_b.vector_dot(self.molecule.Db)
        # L += self.molecule.vks_a.vector_dot(self.input_density_wfn.Da()) + self.molecule.vks_b.vector_dot(self.input_density_wfn.Db())
        return L

    def grad_vext_WuYang_grid(self, v=None):
        """
        grad_a = dL/dvxc_a = - (n_a-n_a_input)
        grad_b = dL/dvxc_b = - (n_b-n_b_input)
        :return:
        """
        if self.four_overlap is None:
            self.four_overlap, _, _, _ = pdft.fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                          self.molecule.basis, self.molecule.mints)
        self.grad_counter += 1

        if v is not None:
            reshaped_vext_app = np.reshape(v, (self.molecule.nbf, self.molecule.nbf))

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijkl,kl->ij", self.four_overlap, reshaped_vext_app) +
                                                self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijkl,kl->ij", self.four_overlap, reshaped_vext_app) +
                                                self.vout_constant * self.molecule.S.np + self.v0_Fock)
            self.molecule.KS_solver(1000, [Vks_a, Vks_b], add_vext=False)
            self.update_vout_constant()

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np

        grad_a = dDa
        grad_b = dDb

        grad = grad_a + grad_b
        grad.shape = self.molecule.nbf**2
        return grad

    def check_gradient_vext_WuYang_grid(self, dv=None):
        if self.v0 != "HartreeLDA":
            print("Changing v0 to Hartree + LDA. Only works for LDA")
            self.get_HartreeLDA_v0()

        if self.four_overlap is None:
            self.four_overlap, _, _, _ = pdft.fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                          self.molecule.basis, self.molecule.mints)

        if self.vext_app is None:
            self.vext_app = np.zeros(self.molecule.nbf**2)

        # reshaped_vext_app = np.reshape(self.vext_app, (self.molecule.nbf, self.molecule.nbf))
        Vks_a = psi4.core.Matrix.from_array(np.einsum("ijkl,kl->ij", self.four_overlap,
                                                      np.reshape(self.vext_app, (self.molecule.nbf, self.molecule.nbf)))
                                            + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(np.einsum("ijkl,kl->ij", self.four_overlap,
                                                      np.reshape(self.vext_app, (self.molecule.nbf, self.molecule.nbf)))
                                            + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        self.molecule.KS_solver(1000, [Vks_a, Vks_b], add_vext=False)
        self.update_vout_constant()

        L = self.Lagrangian_vext_WuYang_grid()
        grad = self.grad_vext_WuYang_grid()

        if dv is None:
            dv = 1e-7 * np.ones_like(self.vext_app)

        grad_app = np.zeros_like(dv)

        print("Start testing. This will time a while...")
        dvsize = dv.shape[0]
        dvsizepercentile = int(dvsize/10)
        for i in range(dvsize):
            if (i%dvsizepercentile) == 0:
                print("%i %% done." %(i/dvsizepercentile*10))
            dvi = np.zeros_like(dv)
            dvi[i] = dv[i]

            L_new = self.Lagrangian_vext_WuYang_grid(dvi+self.vext_app)

            grad_app[i] = (L_new-L) / dvi[i]
        print(np.sum(grad*grad_app)/np.linalg.norm(grad)/np.linalg.norm(grad_app))
        print(np.linalg.norm(grad_app-grad)/np.linalg.norm(grad))
        return grad, grad_app

#%% Section for finding vext_approximate on basis in order to avoid the singularity in real vext. DOES NOT WORK NOW.
    def find_vext_scipy_WuYang(self, maxiter=14000, opt_method="trust-krylov", opt=None,
                              continue_opt=False, find_vxc_grid=True):

        if self.v0 != "HartreeLDAappext":
            print("Changing v0 to Hartree + LDA. Only works for LDA")
            self.get_HartreeLDAappext_v0()

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)
        if not continue_opt:
            print("Zero the old result for a new calculation..")
            self.v_output = np.zeros_like(self.v_output)

            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.KS_solver(100, [Vks_a, Vks_b], add_vext=False)
            self.update_vout_constant()

        print("<<<<<<<<<<<<<<<<<<<<<<WuYang vxc Inversion %s<<<<<<<<<<<<<<<<<<<" % opt_method)

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa + dDb)
        print("|n| before", np.sum(np.abs(dn) * self.molecule.w))
        if opt is None:
            opt = {
                "disp": False,
                "maxiter": maxiter,
                # "eps": 1e-7
                # "norm": 2,
                # "gtol": 1e-7
            }

        result_x = optimizer.minimize(self.Lagrangian_WuYang, self.v_output,
                                      jac=self.grad_WuYang,
                                      hess=self.hess_WuYang,
                                      args=(False),
                                      method=opt_method,
                                      options=opt)
        nbf = int(result_x.x.shape[0] / 2)
        v_output_a = result_x.x[:nbf]
        v_output_b = result_x.x[nbf:]

        Vks_a = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap,
                      v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap,
                      v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        self.molecule.KS_solver(100, [Vks_a, Vks_b], add_vext=False)
        self.update_vout_constant()

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa + dDb)
        print("|jac|", np.linalg.norm(result_x.jac), "|n|", np.sum(np.abs(dn) * self.molecule.w), "L after",
              result_x.fun)
        print("Ts", self.molecule.Da.vector_dot(self.molecule.T) + self.molecule.Db.vector_dot(self.molecule.T))
        print("dTs", np.trace(np.dot(self.input_density_wfn.Da().np + self.input_density_wfn.Db().np -
                                     self.molecule.Da.np - self.molecule.Db.np, self.molecule.T.np)))
        print("eigenA")
        print(self.input_density_wfn.epsilon_a().np[:self.molecule.nalpha])
        print(self.molecule.eig_a.np[:self.molecule.nalpha])
        print("wfnDiff", self.input_density_wfn.Ca().vector_dot(self.molecule.Ca)
              / np.linalg.norm(self.input_density_wfn.Ca().np) / np.linalg.norm(self.molecule.Ca.np))
        print("Constant potential: ", self.vout_constant)

        # Update info
        if find_vxc_grid:
            assert self.v0 == "HartreeLDAappext"

            # # Get vH_mol
            # self.get_mol_vH()
            #
            # nbf = int(result_x.x.shape[0] / 2)
            # v_output_a = result_x.x[:nbf]
            # v_output_b = result_x.x[nbf:]
            #
            # print("Start to get vxc_mol.")
            # self.vp_basis.Vpot.set_D([self.molecule.Da, self.molecule.Db])
            # self.vp_basis.Vpot.properties()[0].set_pointers(self.molecule.Da, self.molecule.Db)
            # print("Pointer setted")
            # mol_vxc_a, mol_vxc_b = pdft.U_xc(self.molecule.Da.np, self.molecule.Db.np, self.vp_basis.Vpot)[-1]
            # print("Doing getting vxc_mol.")

            if self.ortho_basis:
                vext_a_grid = self.vp_basis.to_grid(np.dot(self.vp_basis.A.np, v_output_a))
                vext_b_grid = self.vp_basis.to_grid(np.dot(self.vp_basis.A.np, v_output_b))
            else:
                vext_a_grid = self.vp_basis.to_grid(v_output_a)
                vext_b_grid = self.vp_basis.to_grid(v_output_b)

            # vext_a_grid += self.vH4v0 - self.vH_mol + self.input_vxc_a - mol_vxc_a
            # vext_b_grid += self.vH4v0 - self.vH_mol + self.input_vxc_b - mol_vxc_b
            approximate_vext = np.copy(self.vext)
            approximate_vext[approximate_vext<self.approximate_vext_cutoff] = self.approximate_vext_cutoff

            # vext_a_grid += approximate_vext
            # vext_b_grid += approximate_vext
        return vext_a_grid, vext_b_grid

    def check_gradient_vext_WuYang(self, dv=None):
        if self.v0 != "HartreeLDAappext":
            print("Changing v0 to Hartree + LDA. Only works for LDA")
            self.get_HartreeLDAappext_v0()

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        nbf = int(self.v_output.shape[0] / 2)
        v_output_a = self.v_output[:nbf]
        v_output_b = self.v_output[nbf:]

        Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)

        self.molecule.KS_solver(100, [Vks_a, Vks_b], add_vext=False)
        self.update_vout_constant()

        L = self.Lagrangian_WuYang(fit4vxc=False)
        grad = self.grad_WuYang(fit4vxc=False)

        if dv is None:
            dv = 1e-7*np.ones_like(self.v_output)

        grad_app = np.zeros_like(dv)

        for i in range(dv.shape[0]):
            dvi = np.zeros_like(dv)
            dvi[i] = dv[i]

            L_new = self.Lagrangian_WuYang(dvi+self.v_output, fit4vxc=False)

            grad_app[i] = (L_new-L) / dvi[i]

            # print(L_new, L, i + 1, "out of ", dv.shape[0])

        print(np.sum(grad*grad_app)/np.linalg.norm(grad)/np.linalg.norm(grad_app))
        print(np.linalg.norm(grad_app-grad)/np.linalg.norm(grad))
        return grad, grad_app

    def vH_quadrature(self, grid_info=None):
        """
        Calculating vH using quadrature integral.
        :return:
        """

        if grid_info is None:
            vH = np.zeros_like(self.molecule.w)
            nblocks = self.molecule.Vpot.nblocks()
            points_func = self.molecule.Vpot.properties()[0]
            blocks = None
        else:
            blocks, npoints, points_func = grid_info
            vH = np.zeros(npoints)
            nblocks = len(blocks)
        # First loop over the outer set of blocks
        num_block_ten_percent = int(nblocks / 10)
        w1_old = 0
        print("vxchole quadrature integral starts: ", end="")
        for l_block in range(nblocks):
            # Print out progress
            if num_block_ten_percent != 0 and l_block % num_block_ten_percent == 0:
                print(".", end="")

            # Obtain general grid information
            if blocks is None:
                l_grid = self.molecule.Vpot.get_block(l_block)
                blocks = None
            else:
                l_grid = blocks[l_block]
            l_x = np.array(l_grid.x())
            l_y = np.array(l_grid.y())
            l_z = np.array(l_grid.z())
            l_npoints = l_x.shape[0]

            dvp_l = np.zeros_like(l_x)
            # Loop over the inner set of blocks
            for r_block in range(self.molecule.Vpot.nblocks()):
                r_grid = self.molecule.Vpot.get_block(r_block)
                r_w = np.array(r_grid.w())
                r_x = np.array(r_grid.x())
                r_y = np.array(r_grid.y())
                r_z = np.array(r_grid.z())
                r_npoints = r_w.shape[0]

                points_func.compute_points(r_grid)
                r_lpos = np.array(r_grid.functions_local_to_global())

                # Compute phi!
                r_phi = np.array(points_func.basis_values()["PHI"])[:r_npoints, :r_lpos.shape[0]]

                # Build a local slice of D
                lD2 = self.input_density_wfn.Da().np + self.input_density_wfn.Db().np
                lD2 = lD2[(r_lpos[:, None], r_lpos)]
                # lD2 += self.molecule.Db.np[(r_lpos[:, None], r_lpos)]

                # Copmute block-rho and block-gamma
                rho2 = np.einsum('pm,mn,pn->p', r_phi, lD2, r_phi, optimize=True)

                # Build the distnace matrix
                R2 = (l_x[:, None] - r_x) ** 2
                R2 += (l_y[:, None] - r_y) ** 2
                R2 += (l_z[:, None] - r_z) ** 2
                # R2 += 1e-34
                if np.any(np.isclose(R2, 0.0)):
                    # R2[np.isclose(R2, 0.0, atol=1e-6)] = np.min(R2[~np.isclose(R2, 0.0)])
                    R2[np.isclose(R2, 0.0)] = np.inf
                Rinv = 1/ np.sqrt(R2)

                dvp_l += np.sum(rho2 * Rinv * r_w, axis=1)

            vH[w1_old:w1_old + l_npoints] += dvp_l
            w1_old += l_npoints
        print("\n")
        return vH

    def _vxc_hole_quadrature(self, grid_info=None, atol = 1e-4):
        """
        Calculating v_XC^hole in RKS (15) using quadrature intrgral to test the ability of it.
        :return:
        """
        if self.vxc_hole_WF is not None and grid_info is None:
            return self.vxc_hole_WF

        restricted = psi4.core.get_global_option("REFERENCE") == "RHF"
        support_methods = ["RHF", "UHF", "CIWavefunction"]
        if not self.input_density_wfn.name() in support_methods:
            raise Exception("%s is not supported. Currently only support:"% self.input_density_wfn.name(), support_methods)
        elif self.input_density_wfn.name() == "CIWavefunction" and (not restricted):
            raise Exception("Unrestricted %s is not supported." % self.input_density_wfn.name())
        elif self.input_density_wfn.name() == "CIWavefunction":
            Tau_ijkl = self.input_density_wfn.get_tpdm("SUM", True).np
            D2 = self.input_density_wfn.get_opdm(-1, -1, "SUM", True).np
            C = self.input_density_wfn.Ca()
            Tau_ijkl = contract("pqrs,ip,jq,ur,vs->ijuv", Tau_ijkl, C, C, C, C)
            D2 = C.np @ D2 @ C.np.T
        else:
            D2a = self.input_density_wfn.Da().np
            D2b = self.input_density_wfn.Db().np
            D2 = D2a + D2b

        # assert np.allclose(self.v0_wfn.Da(), self.v0_wfn.Db()), "mRKS currently only supports RHF."

        if grid_info is None:
            vxchole = np.zeros_like(self.molecule.w)
            nblocks = self.molecule.Vpot.nblocks()

            points_func = self.molecule.Vpot.properties()[0]
            points_func.set_deriv(0)

            blocks = None
        else:
            blocks, npoints, points_func = grid_info
            vxchole = np.zeros(npoints)
            nblocks = len(blocks)
            points_func.set_deriv(0)

        if not restricted:
            vxchole_b = np.zeros_like(vxchole)

        # First loop over the outer set of blocks
        num_block_ten_percent = int(nblocks / 10)
        w1_old = 0
        print("vxchole quadrature with %s integral starts (%i points): " % (self.input_density_wfn.name(), self.molecule.w.shape[0]), end="")
        start_time = time.time()
        for l_block in range(nblocks):
            # Print out progress
            if num_block_ten_percent != 0 and l_block % num_block_ten_percent == 0:
                print(".", end="")

            # Obtain general grid information
            if blocks is None:
                l_grid = self.molecule.Vpot.get_block(l_block)
            else:
                l_grid = blocks[l_block]

            l_x = np.array(l_grid.x())
            l_y = np.array(l_grid.y())
            l_z = np.array(l_grid.z())
            l_npoints = l_x.shape[0]

            points_func.compute_points(l_grid)

            l_lpos = np.array(l_grid.functions_local_to_global())
            l_phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :l_lpos.shape[0]]

            if restricted:
                lD1 = D2[(l_lpos[:, None], l_lpos)]
                rho1 = contract('pm,mn,pn->p', l_phi, lD1, l_phi)
                rho1inv = (1 / rho1)[:, None]

            else:
                lD1_a = D2a[(l_lpos[:, None], l_lpos)]
                lD1_b = D2b[(l_lpos[:, None], l_lpos)]
                rho1_a = contract('pm,mn,pn->p', l_phi, lD1_a, l_phi)
                rho1ainv = (1 / rho1_a)[:, None]
                rho1_b = contract('pm,mn,pn->p', l_phi, lD1_b, l_phi)
                rho1binv = (1 / rho1_b)[:, None]


            dvp_l = np.zeros_like(l_x)
            if not restricted:
                dvp_l_b = np.zeros_like(l_x)

            # Loop over the inner set of blocks
            for r_block in range(self.molecule.Vpot.nblocks()):
                r_grid = self.molecule.Vpot.get_block(r_block)
                r_w = np.array(r_grid.w())
                r_x = np.array(r_grid.x())
                r_y = np.array(r_grid.y())
                r_z = np.array(r_grid.z())
                r_npoints = r_w.shape[0]

                points_func.compute_points(r_grid)

                r_lpos = np.array(r_grid.functions_local_to_global())

                # Compute phi!
                r_phi = np.array(points_func.basis_values()["PHI"])[:r_npoints, :r_lpos.shape[0]]

                # Build a local slice of D
                if self.input_density_wfn.name() == "CIWavefunction":
                    lD2 = D2[(r_lpos[:, None], r_lpos)]
                    rho2 = contract('pm,mn,pn->p', r_phi, lD2, r_phi)

                    p,q,r,s = np.meshgrid(l_lpos, l_lpos, r_lpos, r_lpos, indexing="ij")
                    Tap_temp = Tau_ijkl[p,q,r,s]

                    n_xc = contract("mnuv,pm,pn,qu,qv->pq", Tap_temp, l_phi, l_phi, r_phi, r_phi)
                    n_xc *= rho1inv
                    n_xc -= rho2
                elif self.input_density_wfn.name() == "RHF":
                    assert restricted
                    lD2 = self.input_density_wfn.Da().np[(l_lpos[:, None], r_lpos)]
                    n_xc = - 2 * contract("mu,nv,pm,pn,qu,qv->pq", lD2, lD2, l_phi, l_phi, r_phi, r_phi)
                    n_xc *= rho1inv
                elif self.input_density_wfn.name() == "UHF": # This is supposed to be the same as (not restricted)
                    assert not restricted
                    lD2_a = self.input_density_wfn.Da().np[(l_lpos[:, None], r_lpos)]
                    lD2_b = self.input_density_wfn.Db().np[(l_lpos[:, None], r_lpos)]
                    n_xc_a = - contract("mu,nv,pm,pn,qu,qv->pq", lD2_a, lD2_a, l_phi, l_phi, r_phi, r_phi) * rho1ainv
                    n_xc_b = - contract("mu,nv,pm,pn,qu,qv->pq", lD2_b, lD2_b, l_phi, l_phi, r_phi, r_phi) * rho1binv

                # Build the distnace matrix
                R2 = (l_x[:, None] - r_x) ** 2
                R2 += (l_y[:, None] - r_y) ** 2
                R2 += (l_z[:, None] - r_z) ** 2
                # R2 += 1e-34
                if np.any(np.isclose(R2, 0.0, atol=atol)):
                    # R2[np.isclose(R2, 0.0)] = np.min(R2[~np.isclose(R2, 0.0)])
                    R2[np.isclose(R2, 0.0, atol=atol)] = np.inf
                Rinv = 1 / np.sqrt(R2)

                if restricted:
                    dvp_l += np.sum(n_xc * Rinv * r_w, axis=1)
                else:
                    dvp_l += np.sum(n_xc_a * Rinv * r_w, axis=1)
                    dvp_l_b += np.sum(n_xc_b * Rinv * r_w, axis=1)

            if restricted:
                vxchole[w1_old:w1_old + l_npoints] += dvp_l
            else:
                vxchole[w1_old:w1_old + l_npoints] += dvp_l
                vxchole_b[w1_old:w1_old + l_npoints] += dvp_l_b
            w1_old += l_npoints

        print("\n")
        print("Totally %i grid points takes %.2fs with max %i points in a block."
              % (vxchole.shape[0], time.time() - start_time, psi4.core.get_global_option("DFT_BLOCK_MAX_POINTS")))
        assert w1_old == vxchole.shape[0], "Somehow the whole space is not fully integrated."
        if blocks is None:
            if restricted:
                self.vxc_hole_WF = vxchole
            else:
                self.vxc_hole_WF = (vxchole, vxchole_b)
            return self.vxc_hole_WF
        else:
            if restricted:
               return vxchole
            else:
                return (vxchole, vxchole_b)

    def _average_local_orbital_energy(self, D, C, eig, Db=None, Cb=None, eig_b=None, grid_info=None):
        """
        (4)(6) in mRKS.
        """
        

        # Nalpha = self.molecule.nalpha
        # Nbeta = self.molecule.nbeta

        if grid_info is None:
            e_bar = np.zeros_like(self.molecule.w)
            nblocks = self.molecule.Vpot.nblocks()

            points_func = self.molecule.Vpot.properties()[0]
            points_func.set_deriv(0)
            blocks = None
        else:
            blocks, npoints, points_func = grid_info
            e_bar = np.zeros(npoints)
            nblocks = len(blocks)

            points_func.set_deriv(0)

        # For unrestricted
        if Db is not None:
            e_bar_beta = np.zeros_like(e_bar)
        iw = 0
        for l_block in range(nblocks):
            # Obtain general grid information
            if blocks is None:
                l_grid = self.molecule.Vpot.get_block(l_block)
            else:
                l_grid = blocks[l_block]
            l_npoints = l_grid.npoints()

            points_func.compute_points(l_grid)
            l_lpos = np.array(l_grid.functions_local_to_global())
            l_phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :l_lpos.shape[0]]
            lD = D[(l_lpos[:, None], l_lpos)]
            lC = C[l_lpos, :]
            rho = np.einsum('pm,mn,pn->p', l_phi, lD, l_phi, optimize=True)
            e_bar[iw:iw+l_npoints] = np.einsum("pm,mi,ni,i,pn->p", l_phi, lC, lC, eig, l_phi, optimize=True) / rho

            if Db is not None:
                lD = Db[(l_lpos[:, None], l_lpos)]
                lC = Cb[l_lpos, :]
                rho = np.einsum('pm,mn,pn->p', l_phi, lD, l_phi, optimize=True)
                e_bar_beta[iw:iw + l_npoints] = np.einsum("pm,mi,ni,i,pn->p", l_phi, lC, lC, eig_b, l_phi, optimize=True) / rho

            iw += l_npoints
        assert iw == e_bar.shape[0], "Somehow the whole space is not fully integrated."
        if Db is None:
            return e_bar
        else:
            return (e_bar, e_bar_beta)

    def _pauli_kinetic_energy_density(self, D, C, occ=None, Db=None, Cb=None, occb=None, grid_info=None):
        """
        (16)(18) in mRKS. But notice this does not return taup but taup/n
        :return:
        """

        if occ is None:
            occ = np.ones(C.shape[1])
        if occb is None and (Db is not None):
            occb = np.ones(Cb.shape[1])

        if grid_info is None:
            taup_rho = np.zeros_like(self.molecule.w)
            nblocks = self.molecule.Vpot.nblocks()

            points_func = self.molecule.Vpot.properties()[0]
            points_func.set_deriv(1)
            blocks = None

        else:
            blocks, npoints, points_func = grid_info
            taup_rho = np.zeros(npoints)
            nblocks = len(blocks)

            points_func.set_deriv(1)

        if Db is not None:
            taup_rho_beta = np.zeros_like(taup_rho)
        iw = 0
        for l_block in range(nblocks):
            # Obtain general grid information
            if blocks is None:
                l_grid = self.molecule.Vpot.get_block(l_block)
            else:
                l_grid = blocks[l_block]
            l_npoints = l_grid.npoints()

            points_func.compute_points(l_grid)
            l_lpos = np.array(l_grid.functions_local_to_global())
            l_phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_x = np.array(points_func.basis_values()["PHI_X"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_y = np.array(points_func.basis_values()["PHI_Y"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_z = np.array(points_func.basis_values()["PHI_Z"])[:l_npoints, :l_lpos.shape[0]]

            lD = D[(l_lpos[:, None], l_lpos)]

            rho = np.einsum('pm,mn,pn->p', l_phi, lD, l_phi, optimize=True)

            lC = C[l_lpos, :]
            # Matrix Methods
            part_x = np.einsum('pm,mi,nj,pn->ijp', l_phi, lC, lC, l_phi_x, optimize=True)
            part_y = np.einsum('pm,mi,nj,pn->ijp', l_phi, lC, lC, l_phi_y, optimize=True)
            part_z = np.einsum('pm,mi,nj,pn->ijp', l_phi, lC, lC, l_phi_z, optimize=True)
            part1_x = (part_x - np.transpose(part_x, (1, 0, 2))) ** 2
            part1_y = (part_y - np.transpose(part_y, (1, 0, 2))) ** 2
            part1_z = (part_z - np.transpose(part_z, (1, 0, 2))) ** 2


            # part1_x = np.triu(part1_x.T, k=1) ** 2
            # part1_y = np.triu(part1_y.T, k=1) ** 2
            # part1_z = np.triu(part1_z.T, k=1) ** 2

            occ_matrix = np.expand_dims(occ, axis=1) @ np.expand_dims(occ, axis=0)

            taup = np.sum((part1_x + part1_y + part1_z).T * occ_matrix, axis=(1,2)) * 0.5

            # Loop Methods, slow but memory friendly
            # taup = np.zeros(l_npoints)
            # for j in range(lC.shape[1]):
            #     for i in range(j):
            #         phiigradj_x = np.einsum('pm,m,n,pn->p', l_phi, lC[:, i], lC[:, j], l_phi_x, optimize=True)
            #         phijgradi_x = np.einsum('pm,m,n,pn->p', l_phi, lC[:, j], lC[:, i], l_phi_x, optimize=True)
            #
            #         phiigradj_y = np.einsum('pm,m,n,pn->p', l_phi, lC[:, i], lC[:, j], l_phi_y, optimize=True)
            #         phijgradi_y = np.einsum('pm,m,n,pn->p', l_phi, lC[:, j], lC[:, i], l_phi_y, optimize=True)
            #
            #         phiigradj_z = np.einsum('pm,m,n,pn->p', l_phi, lC[:, i], lC[:, j], l_phi_z, optimize=True)
            #         phijgradi_z = np.einsum('pm,m,n,pn->p', l_phi, lC[:, j], lC[:, i], l_phi_z, optimize=True)
            #         taup += ((phiigradj_x - phijgradi_x) ** 2 + (phiigradj_y - phijgradi_y) ** 2 + (
            #                 phiigradj_z - phijgradi_z) ** 2) * occ[i] * occ[j]

            taup_rho[iw:iw + l_npoints] = taup / rho ** 2 * 0.5

            if Db is not None:
                lD = Db[(l_lpos[:, None], l_lpos)]
                rho = np.einsum('pm,mn,pn->p', l_phi, lD, l_phi, optimize=True)
                lC = Cb[l_lpos, :]
                part_x = np.einsum('pm,mi,nj,pn->ijp', l_phi, lC, lC, l_phi_x, optimize=True)
                part_y = np.einsum('pm,mi,nj,pn->ijp', l_phi, lC, lC, l_phi_y, optimize=True)
                part_z = np.einsum('pm,mi,nj,pn->ijp', l_phi, lC, lC, l_phi_z, optimize=True)
                part1_x = (part_x - np.transpose(part_x, (1, 0, 2))) ** 2
                part1_y = (part_y - np.transpose(part_y, (1, 0, 2))) ** 2
                part1_z = (part_z - np.transpose(part_z, (1, 0, 2))) ** 2

                # part1_x = np.triu(part1_x.T, k=1) ** 2
                # part1_y = np.triu(part1_y.T, k=1) ** 2
                # part1_z = np.triu(part1_z.T, k=1) ** 2

                occb_matrix = np.expand_dims(occb, axis=1) @ np.expand_dims(occb, axis=0)

                taup_beta = np.sum((part1_x + part1_y + part1_z).T * occb_matrix, axis=(1, 2)) * 0.5

                taup_rho_beta[iw:iw + l_npoints] = taup_beta / rho ** 2 * 0.5

            iw += l_npoints
        assert iw == taup_rho.shape[0], "Somehow the whole space is not fully integrated."
        if Db is None:
            return taup_rho
        else:
            return (taup_rho, taup_rho_beta)

    def _modified_pauli_kinetic_energy_density(self, D, C, occ=None, Db=None, Cb=None, occb=None, grid_info=None):
        """
        (16)(18) in mRKS. But notice this does not return taup but taup/n
        :return:
        """

        if occ is None:
            occ = np.ones(C.shape[1])
        if occb is None and (Db is not None):
            occb = np.ones(Cb.shape[1])
        if grid_info is None:
            taup_rho = np.zeros_like(self.molecule.w)
            nblocks = self.molecule.Vpot.nblocks()

            points_func = self.molecule.Vpot.properties()[0]
            points_func.set_deriv(1)
            blocks = None
        else:
            blocks, npoints, points_func = grid_info
            taup_rho = np.zeros(npoints)
            nblocks = len(blocks)

            points_func.set_deriv(1)
        if Db is not None:
            taup_rho_beta = np.zeros_like(taup_rho)
        iw = 0
        for l_block in range(nblocks):
            # Obtain general grid information
            if blocks is None:
                l_grid = self.molecule.Vpot.get_block(l_block)
            else:
                l_grid = blocks[l_block]
            l_npoints = l_grid.npoints()
            points_func.compute_points(l_grid)
            l_lpos = np.array(l_grid.functions_local_to_global())
            l_phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_x = np.array(points_func.basis_values()["PHI_X"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_y = np.array(points_func.basis_values()["PHI_Y"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_z = np.array(points_func.basis_values()["PHI_Z"])[:l_npoints, :l_lpos.shape[0]]
            lD = D[(l_lpos[:, None], l_lpos)]
            rho = np.einsum('pm,mn,pn->p', l_phi, lD, l_phi, optimize=True)
            lC = C[l_lpos, :]
            # Matrix Methods
            part_x = np.einsum('pm,mi,nj,pn->ijp', l_phi_x, lC, lC, l_phi_x, optimize=True)
            part_y = np.einsum('pm,mi,nj,pn->ijp', l_phi_y, lC, lC, l_phi_y, optimize=True)
            part_z = np.einsum('pm,mi,nj,pn->ijp', l_phi_z, lC, lC, l_phi_z, optimize=True)
            phi_iphi_j = np.einsum('pm,mi,nj,pn->ijp', l_phi, lC, lC, l_phi, optimize=True) * (part_x + part_y + part_z)

            occ_matrix = np.expand_dims(occ, axis=1) @ np.expand_dims(occ, axis=0)
            taup = -np.sum(phi_iphi_j.T * occ_matrix, axis=(1, 2))
            taup_rho[iw:iw + l_npoints] = taup / rho ** 2
            if Db is not None:
                lD = Db[(l_lpos[:, None], l_lpos)]
                rho = np.einsum('pm,mn,pn->p', l_phi, lD, l_phi, optimize=True)
                lC = Cb[l_lpos, :]
                part_x = np.einsum('pm,mi,nj,pn->ijp', l_phi_x, lC, lC, l_phi_x, optimize=True)
                part_y = np.einsum('pm,mi,nj,pn->ijp', l_phi_y, lC, lC, l_phi_y, optimize=True)
                part_z = np.einsum('pm,mi,nj,pn->ijp', l_phi_z, lC, lC, l_phi_z, optimize=True)
                phi_iphi_j = np.einsum('pm,mi,nj,pn->ijp', l_phi, lC, lC, l_phi, optimize=True) * (part_x + part_y + part_z)
                occb_matrix = np.expand_dims(occb, axis=1) @ np.expand_dims(occb, axis=0)
                taup_beta = -np.sum(phi_iphi_j.T * occb_matrix, axis=(1, 2))
                taup_rho_beta[iw:iw + l_npoints] = taup_beta / rho ** 2
            iw += l_npoints
        assert iw == taup_rho.shape[0], "Somehow the whole space is not fully integrated."
        if Db is None:
            return taup_rho
        else:
            return (taup_rho, taup_rho_beta)

    def _restricted_mRKS(self, maxiter, vxc_grid, scf_maxiter, v_tol, D_tol, eig_tol, frac_old, WF_method, init):
        Nalpha = self.molecule.nalpha

        if self.v0 != "Hartree":
            self.change_v0("Hartree")

        # Preparing for WF
        if WF_method == "CIWavefunction":
            # Solving for Generalized Fock (GFM)>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            opdm = self.input_density_wfn.get_opdm(-1,-1,"SUM",False).np
            T = self.input_density_wfn.get_tpdm("SUM", True).np
            Ca = self.input_density_wfn.Ca().np

            # ERI Memory check
            nbf = self.molecule.S.shape[0]
            I_size = (nbf ** 4) * 8.e-9 * 2
            numpy_memory = 2
            print('Size of the ERI tensor and 2-particle DM will be %4.2f GB.' % (I_size))
            memory_footprint = I_size * 1.5
            if I_size > numpy_memory:
                psi4.core.clean()
                raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted memory \
                                 limit of %4.2f GB." % (memory_footprint, numpy_memory))
            else:
                print("Memory taken by ERI integral matrix and 2pdm is: %.3f" %memory_footprint)
            # Compute AO-basis ERIs
            I = self.molecule.mints.ao_eri()
            # Transfer the AO ERI into MO ERI
            I = np.einsum("ijkl,ip,jq,kr,ls", I, Ca, Ca, Ca, Ca, optimize=True)
            I = 0.5 * I + 0.25 * np.transpose(I, [0, 1, 3, 2]) + 0.25 * np.transpose(I, [1, 0, 2, 3])
            # Transfer the AO H into MO h
            h = Ca.T @ self.molecule.H.np @ Ca

            # F is contructed on the basis of MOs, which are orthonormal
            # F_GFM = opdm @ h + np.einsum("aqrs,pqrs->ap", I, T, optimize=True)
            F_GFM = opdm @ h + np.einsum("rsnq,rsmq->mn", I, T, optimize=True)
            F_GFM = 0.5 * (F_GFM + F_GFM.T)

            nbf = self.molecule.nbf
            C_a_GFM = psi4.core.Matrix(nbf, nbf)
            eigs_a_GFM = psi4.core.Vector(nbf)
            psi4.core.Matrix.from_array(F_GFM).diagonalize(C_a_GFM, eigs_a_GFM, psi4.core.DiagonalizeOrder.Ascending)

            eigs_a_GFM = eigs_a_GFM.np / 2.0  # RHF
            C_a_GFM = C_a_GFM.np
            # Transfer to AOs
            C_a_GFM = Ca @ C_a_GFM
            print("CIWavefunction GFM eigenvalues:", eigs_a_GFM)

            del T, I
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # ============================================================================================
            # Solving for Natural Orbitals (NO)>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            C_a_NO = psi4.core.Matrix(nbf, nbf)
            eigs_a_NO = psi4.core.Vector(nbf)
            psi4.core.Matrix.from_array(opdm).diagonalize(C_a_NO, eigs_a_NO, psi4.core.DiagonalizeOrder.Descending)

            eigs_a_NO = eigs_a_NO.np / 2
            C_a_NO = C_a_NO.np
            C_a_NO_AO = Ca @ C_a_NO
            print("CIWavefunction Occupation Number:", eigs_a_NO)

            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            ebarWF = self._average_local_orbital_energy(self.input_density_wfn.Da().np, C_a_GFM, eigs_a_GFM)
            # ebarWF = self._average_local_orbital_energy(self.input_density_wfn.Da().np,
            #                                            self.input_density_wfn.Ca().np[:,:Nalpha], self.input_density_wfn.epsilon_a().np[:Nalpha])
            taup_rho_WF = self._pauli_kinetic_energy_density(self.input_density_wfn.Da().np, C_a_NO_AO, eigs_a_NO)
            # emax = eigs_a_GFM[self.molecule.nalpha-1]
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        elif WF_method == "RHF":
            ebarWF = self._average_local_orbital_energy(self.input_density_wfn.Da().np,
                                                       self.input_density_wfn.Ca().np[:,:Nalpha], self.input_density_wfn.epsilon_a().np[:Nalpha])
            taup_rho_WF = self._pauli_kinetic_energy_density(self.input_density_wfn.Da().np, self.input_density_wfn.Ca().np[:,:Nalpha])

        # I am not sure about this one.
        # emax = self.input_density_wfn.epsilon_a().np[self.molecule.nalpha-1]
        emax = np.max(ebarWF)

        vxchole = self._vxc_hole_quadrature()

        # initial calculation:
        if init is None:
            self.molecule.KS_solver(scf_maxiter, V=None)
        elif init == "continue":
            assert self.molecule.Da is not None
        elif init == "LDA":
            self.molecule.scf(scf_maxiter)

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("\n|n| init:", np.sum(np.abs(dn)*self.molecule.w) / (2 * Nalpha))

        vxc_old = 0.0
        Da_old = 0.0
        eig_old = 0.0
        for mRKS_step in range(1, maxiter+1):
            assert np.allclose(self.molecule.Da, self.molecule.Db), "mRKS currently only supports RHF."

            # ebarKS = self._average_local_orbital_energy(self.molecule.Da.np, self.molecule.Ca.np[:,:Nalpha], self.molecule.eig_a.np[:Nalpha] + self.vout_constant)
            ebarKS = self._average_local_orbital_energy(self.molecule.Da.np, self.molecule.Ca.np[:,:Nalpha], self.molecule.eig_a.np[:Nalpha])
            taup_rho_KS = self._pauli_kinetic_energy_density(self.molecule.Da.np, self.molecule.Ca.np[:,:Nalpha])

            # self.vout_constant = emax - self.molecule.eig_a.np[self.molecule.nalpha - 1]
            potential_shift = emax - np.max(ebarKS)
            self.vout_constant = potential_shift

            vxc = vxchole + ebarKS - ebarWF + taup_rho_WF - taup_rho_KS + potential_shift

            # Add compulsory mixing parameter close to the convergence to help convergence HOPEFULLY
            # Check vp convergence
            if np.sum(np.abs(vxc - vxc_old) * self.molecule.w) < v_tol:
                print("vxc stops updating.")
                break
            elif mRKS_step != 1:
                vxc = vxc * (1 - frac_old) + vxc_old * frac_old

            vxc_Fock = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(vxc) + self.v0_Fock)
            # vxc_Fock = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(vxc))
            self.molecule.KS_solver(100, V=[vxc_Fock, vxc_Fock])
            # self.molecule.scf(scf_maxiter, vp_matrix=[vxc_Fock, vxc_Fock], vxc_flag=False)

            dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
            dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
            dn = self.molecule.to_grid(dDa + dDb)
            print("Iter: %i,  |n|: %.4e"%(mRKS_step, np.sum(np.abs(dn) * self.molecule.w) / (2 * Nalpha)))

            # Another convergence criteria, DM
            if ((np.linalg.norm(self.molecule.Da.np - Da_old) / self.molecule.Da.np.shape[0] ** 2) < D_tol) and \
                    ((np.linalg.norm(self.molecule.eig_a.np - eig_old) / self.molecule.eig_a.np.shape[0]) < eig_tol):
                print("KSDFT stops updating.")
                break

            vxc_old = vxc
            Da_old = np.copy(self.molecule.Da)
            eig_old = np.copy(self.molecule.eig_a)

        if vxc_grid is not None:
            grid_info = self.get_blocks_from_grid(vxc_grid)
            vxchole = self._vxc_hole_quadrature(blocks=grid_info)
            if WF_method == "CIWavefunction":
                ebarWF = self._average_local_orbital_energy(self.input_density_wfn.Da().np, C_a_GFM, eigs_a_GFM, blocks=grid_info)
                taup_rho_WF = self._pauli_kinetic_energy_density(self.input_density_wfn.Da().np, C_a_NO_AO, eigs_a_NO, blocks=grid_info)
            elif WF_method == "RHF":
                ebarWF = self._average_local_orbital_energy(self.input_density_wfn.Da().np,
                                                            self.input_density_wfn.Ca().np[:, :Nalpha],
                                                            self.input_density_wfn.epsilon_a().np[:Nalpha], blocks=grid_info)
                taup_rho_WF = self._pauli_kinetic_energy_density(self.input_density_wfn.Da().np,
                                                                 self.input_density_wfn.Ca().np[:, :Nalpha], blocks=grid_info)
            ebarKS = self._average_local_orbital_energy(self.molecule.Da.np, self.molecule.Ca.np[:,:Nalpha],
                                                        self.molecule.eig_a.np[:Nalpha], blocks=grid_info)
            taup_rho_KS = self._pauli_kinetic_energy_density(self.molecule.Da.np, self.molecule.Ca.np[:,:Nalpha],
                                                             blocks=grid_info)

            potential_shift = np.max(ebarWF) - np.max(ebarKS)
            self.vout_constant = potential_shift

            vxc = vxchole + ebarKS - ebarWF + taup_rho_WF - taup_rho_KS + potential_shift

        return vxc, vxchole, ebarKS, ebarWF, taup_rho_WF, taup_rho_KS

    def _unrestricted_mRKS(self, maxiter, vxc_grid, scf_maxiter, v_tol, D_tol, eig_tol, frac_old, WF_method, init, _no_tau_apprx=False):
        """

        :param maxiter:
        :param vxc_grid:
        :param scf_maxiter:
        :param v_tol:
        :param D_tol:
        :param eig_tol:
        :param frac_old:
        :param WF_method:
        :param _no_tau_apprx: For Unrestricted cases, sometimes tau WF is bad. dtau is usually small so ignoring it is one approximation.
        :return:
        """
        assert WF_method=="UHF", "HF is the only WF method that currently supports unrestricted mRKS."
        Nalpha = self.molecule.nalpha
        Nbeta = self.molecule.nbeta

        if self.v0 != "Hartree":
            self.change_v0("Hartree")

        # Preparing for WF
        ebarWF_a, ebarWF_b = self._average_local_orbital_energy(self.input_density_wfn.Da().np,
                                                             self.input_density_wfn.Ca().np[:,:Nalpha],
                                                             self.input_density_wfn.epsilon_a().np[:Nalpha],
                                                             Db=self.input_density_wfn.Db().np,
                                                             Cb=self.input_density_wfn.Cb().np[:,:Nbeta],
                                                             eig_b=self.input_density_wfn.epsilon_b().np[:Nbeta])

        taup_rho_WF_a, taup_rho_WF_b = self._pauli_kinetic_energy_density(self.input_density_wfn.Da().np,
                                                                          self.input_density_wfn.Ca().np[:,:Nalpha],
                                                                          Db=self.input_density_wfn.Db().np,
                                                                          Cb=self.input_density_wfn.Cb().np[:, :Nbeta],
                                                                          )

        # I am not sure about this one.
        # emax = self.input_density_wfn.epsilon_a().np[self.molecule.nalpha-1]
        emax_a = np.max(ebarWF_a)
        emax_b = np.max(ebarWF_b)

        vxchole_a, vxchole_b = self._vxc_hole_quadrature()

        # initial calculation:
        # initial calculation:
        if init is None:
            self.molecule.KS_solver(scf_maxiter, V=None)
        elif init == "continue":
            assert self.molecule.Da is not None
        elif init == "LDA":
            self.molecule.scf(scf_maxiter)

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dna = self.molecule.to_grid(dDa)
        dnb = self.molecule.to_grid(dDb)
        print("\n|n| init", np.sum(np.abs(dna)*self.molecule.w) / Nalpha, np.sum(np.abs(dnb)*self.molecule.w) / Nbeta)

        vxc_old_a = 0.0
        vxc_old_b = 0.0
        Da_old = 0.0
        eig_old = 0.0
        for mRKS_step in range(1, maxiter+1):
            ebarKS_a, ebarKS_b = self._average_local_orbital_energy(self.molecule.Da.np,
                                                                 self.molecule.Ca.np[:,:Nalpha],
                                                                 self.molecule.eig_a.np[:Nalpha],
                                                                 Db=self.molecule.Db.np,
                                                                 Cb=self.molecule.Cb.np[:, :Nbeta],
                                                                 eig_b=self.molecule.eig_b.np[:Nbeta]
                                                                 )

            taup_rho_KS_a, taup_rho_KS_b = self._pauli_kinetic_energy_density(self.molecule.Da.np,
                                                                           self.molecule.Ca.np[:,:Nalpha],
                                                                           Db=self.molecule.Db.np,
                                                                           Cb=self.molecule.Cb.np[:, :Nbeta],
                                                                           )
            potential_shift_a = emax_a - np.max(ebarKS_a)
            potential_shift_b = emax_b - np.max(ebarKS_b)
            self.vout_constant = (potential_shift_a, potential_shift_b)
            # if mRKS_step == 1:
            #     vxc_a = vxchole_a + potential_shift_a
            #     vxc_b = vxchole_b + potential_shift_b
            # elif mRKS_step == 2:
            #     vxc_a = vxchole_a + ebarKS_a - ebarWF_a + potential_shift_a
            #     vxc_b = vxchole_b + ebarKS_b - ebarWF_b + potential_shift_b
            # else:
            vxc_a = vxchole_a + ebarKS_a - ebarWF_a + taup_rho_WF_a - taup_rho_KS_a + potential_shift_a
            vxc_b = vxchole_b + ebarKS_b - ebarWF_b + taup_rho_WF_b - taup_rho_KS_b + potential_shift_b
            if _no_tau_apprx:
                vxc_a -= taup_rho_WF_a - taup_rho_KS_a
                vxc_b -= taup_rho_WF_b - taup_rho_KS_b

            # Add compulsory mixing parameter close to the convergence to help convergence HOPEFULLY
            # Check vp convergence
            if np.sum(np.abs(vxc_a - vxc_old_a) * self.molecule.w) < v_tol and np.sum(
                    np.abs(vxc_b - vxc_old_b) * self.molecule.w) < v_tol:
                print("vxc stops updating.")
                break
            elif mRKS_step != 1:
                vxc_a = vxc_a * (1 - frac_old) + vxc_old_a * frac_old
                vxc_b = vxc_b * (1 - frac_old) + vxc_old_b * frac_old

            # vxc_Fock = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(vxc) + self.v0_Fock)mol.nal
            vxc_Fock_a = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(vxc_a))
            vxc_Fock_b = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(vxc_b))

            if Nbeta==0:  # For systems w/ 1 electron.
                vxc_Fock_b.np[:] = 0

            # The idea is (though not sure if it helps.), for steps smaller than 5, use input density vH to guide.
            # After that, use its's self vH.
            # if mRKS_step < 5:
            #     vxc_Fock_a.np[:] += self.v0_Fock
            #     vxc_Fock_b.np[:] += self.v0_Fock
            #     self.molecule.KS_solver(100, V=[vxc_Fock_a, vxc_Fock_b])
            # else:
            #     self.molecule.scf(scf_maxiter, vp_matrix=[vxc_Fock_a, vxc_Fock_b], vxc_flag=False)
            vxc_Fock_a.np[:] += self.v0_Fock
            vxc_Fock_b.np[:] += self.v0_Fock
            self.molecule.KS_solver(100, V=[vxc_Fock_a, vxc_Fock_b])

            dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
            dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
            dna = self.molecule.to_grid(dDa)
            dnb = self.molecule.to_grid(dDb)
            print("Iter: %i,  |n|: %.4e %.4e" % (
            mRKS_step, np.sum(np.abs(dna) * self.molecule.w) / Nalpha, np.sum(np.abs(dnb) * self.molecule.w) / Nbeta))

            # Another convergence criteria, DM
            if ((np.linalg.norm(self.molecule.Da.np - Da_old) / self.molecule.Da.np.shape[0] ** 2) < D_tol) and \
                    ((np.linalg.norm(self.molecule.eig_a.np - eig_old) / self.molecule.eig_a.np.shape[0]) < eig_tol):
                print("KSDFT stops updating.")
                break

            vxc_old_a = vxc_a
            vxc_old_b = vxc_b
            Da_old = np.copy(self.molecule.Da)
            eig_old = np.copy(self.molecule.eig_a)

        if vxc_grid is not None:
            grid_info = self.get_blocks_from_grid(vxc_grid)
            vxchole_a, vxchole_b = self._vxc_hole_quadrature(blocks=grid_info)

            ebarWF_a, ebarWF_b = self._average_local_orbital_energy(self.input_density_wfn.Da().np,
                                                                    self.input_density_wfn.Ca().np[:, :Nalpha],
                                                                    self.input_density_wfn.epsilon_a().np[:Nalpha],
                                                                    Db=self.input_density_wfn.Db().np,
                                                                    Cb=self.input_density_wfn.Cb().np[:, :Nbeta],
                                                                    eig_b=self.input_density_wfn.epsilon_b().np[:Nbeta],
                                                                    blocks=grid_info)
            taup_rho_WF_a, taup_rho_WF_b = self._pauli_kinetic_energy_density(self.input_density_wfn.Da().np,
                                                                              self.input_density_wfn.Ca().np[:,
                                                                              :Nalpha],
                                                                              Db=self.input_density_wfn.Db().np,
                                                                              Cb=self.input_density_wfn.Cb().np[:,
                                                                                 :Nbeta],
                                                                              blocks=grid_info)
            ebarKS_a, ebarKS_b = self._average_local_orbital_energy(self.molecule.Da.np,
                                                                 self.molecule.Ca.np[:,:Nalpha],
                                                                 self.molecule.eig_a.np[:Nalpha],
                                                                 Db=self.molecule.Db.np,
                                                                 Cb=self.molecule.Cb.np[:, :Nbeta],
                                                                 eig_b=self.molecule.eig_b.np[:Nbeta],
                                                                 blocks=grid_info)
            taup_rho_KS_a, taup_rho_KS_b = self._pauli_kinetic_energy_density(self.molecule.Da.np,
                                                                           self.molecule.Ca.np[:,:Nalpha],
                                                                           Db=self.molecule.Db.np,
                                                                           Cb=self.molecule.Cb.np[:, :Nbeta],
                                                                           blocks=grid_info)
            potential_shift_a = np.max(ebarWF_a) - np.max(ebarKS_a)
            potential_shift_b = np.max(ebarWF_b) - np.max(ebarKS_b)
            self.vout_constant = (potential_shift_a, potential_shift_b)
            vxc_a = vxchole_a + ebarKS_a - ebarWF_a + taup_rho_WF_a - taup_rho_KS_a + potential_shift_a
            vxc_b = vxchole_b + ebarKS_b - ebarWF_b + taup_rho_WF_b - taup_rho_KS_b + potential_shift_b
            if _no_tau_apprx:
                vxc_a -= taup_rho_WF_a - taup_rho_KS_a
                vxc_b -= taup_rho_WF_b - taup_rho_KS_b
        return (vxc_a, vxc_b), (vxchole_a, vxchole_b), (ebarKS_a, ebarKS_b), (ebarWF_a, ebarWF_b), \
               (taup_rho_WF_a, taup_rho_WF_b), (taup_rho_KS_a, taup_rho_KS_b)

    def mRKS(self, maxiter=21, vxc_grid=None, scf_maxiter=300,
             v_tol=1e-3, D_tol=1e-7, eig_tol=1e-7, frac_old=0, init="LDA"):
        """
        Get vxc on the grid with mRKS methods. Only works for RHF right now.
        
        init: {"LDA", None, "continue"}. Initial guess as the starting point.
        """
        support_methods = ["RHF", "UHF", "CIWavefunction"]
        restricted = psi4.core.get_global_option("REFERENCE") == "RHF"

        if not (self.input_density_wfn.name() in support_methods):
            raise Exception("%s is not supported. Currently only support:"%self.input_density_wfn.name(), support_methods)
        elif self.input_density_wfn.name() == "CIWavefunction" and (not restricted):
            raise Exception("Unrestricted %s is not supported."% self.input_density_wfn.name())

        if restricted:
            vxc, vxchole, ebarKS, ebarWF, \
            taup_rho_WF, taup_rho_KS = self._restricted_mRKS(maxiter, vxc_grid, scf_maxiter,
                                                                                           v_tol, D_tol, eig_tol, frac_old,
                                                                                           self.input_density_wfn.name(), init)
        else:
            vxc, vxchole, ebarKS, ebarWF, \
            taup_rho_WF, taup_rho_KS = self._unrestricted_mRKS(maxiter, vxc_grid, scf_maxiter,
                                                               v_tol, D_tol, eig_tol, frac_old,
                                                               self.input_density_wfn.name(), init)

        return vxc, vxchole, ebarKS, ebarWF, taup_rho_WF, taup_rho_KS

    def get_blocks_from_grid(self, grid):
        """
        Return blocks for a given grid.
        """
        assert (grid.shape[0] == 3) or (grid.shape[0] == 4)

        epsilon = psi4.core.get_global_option("CUBIC_BASIS_TOLERANCE") * 1e-5
        extens = psi4.core.BasisExtents(self.molecule.wfn.basisset(), epsilon)
        max_points = psi4.core.get_global_option("DFT_BLOCK_MAX_POINTS") - 1
        # max_points = self.molecule.Vpot.properties()[0].max_points()
        
        
        if_w = (grid.shape[0] == 4)

        npoints = grid.shape[1]

        nblocks = int(np.floor(npoints/max_points))

        blocks = []
        
        max_functions = 0
        idx = 0
        for nb in range(nblocks):
            x = psi4.core.Vector.from_array(grid[0][idx:idx+max_points])
            y = psi4.core.Vector.from_array(grid[1][idx:idx+max_points])
            z = psi4.core.Vector.from_array(grid[2][idx:idx+max_points])
            if if_w:
                w = psi4.core.Vector.from_array(grid[3][idx: idx + max_points])
            else:
                w = psi4.core.Vector.from_array(np.zeros_like(grid[2][idx:idx+max_points]))  # When w is not necessary and not given

            blocks.append(psi4.core.BlockOPoints(x, y, z, w, extens))
            max_functions = max_functions if max_functions > len(blocks[-1].functions_local_to_global()) else len(
                blocks[-1].functions_local_to_global())
            
            idx += max_points
            max_functions = max_functions if max_functions > len(blocks[-1].functions_local_to_global()) else len(
                blocks[-1].functions_local_to_global())
        # One more un-full block
        if idx < npoints:
            x = psi4.core.Vector.from_array(grid[0][idx:])
            y = psi4.core.Vector.from_array(grid[1][idx:])
            z = psi4.core.Vector.from_array(grid[2][idx:])
            if if_w:
                w = psi4.core.Vector.from_array(grid[3][idx:])
            else:
                w = psi4.core.Vector.from_array(np.zeros_like(grid[2][idx:]))  # When w is not necessary and not given
            blocks.append(psi4.core.BlockOPoints(x, y, z, w, extens))
            max_functions = max_functions if max_functions > len(blocks[-1].functions_local_to_global()) else len(
                blocks[-1].functions_local_to_global())

        point_func = psi4.core.RKSFunctions(self.input_density_wfn.basisset(), max_points, max_functions)
        point_func.set_pointers(self.input_density_wfn.Da())
        return blocks, npoints, point_func