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


if __name__ == "__main__":
    psi4.set_num_threads(2)


class Molecule(pdft.U_Molecule):
    def __init__(self, geometry, basis, method, omega=1, mints=None, jk=None):
        super().__init__(geometry, basis, method, omega=omega, mints=mints, jk=jk)

    def scf_inversion(self, maxiter, V=None, print_energies=False, vp_matrix=None, add_vext=True):
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

        Vks_a = V_a
        Vks_b = V_b
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
        self.get_esp4v0()

        # v_ext
        self.vext = None
        self.approximate_vext_cutoff = None  # Used as initial guess for vext calculation to cut the singularity.
        self.vH4v0 = None
        self.v0_Fock = None
        self.get_vH_vext()
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
            print("no xc")
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

    def get_input_vxc(self):
        self.input_vxc_a, self.input_vxc_b = pdft.U_xc(self.input_density_wfn.Da().np, self.input_density_wfn.Db().np,
                                                       # self.molecule.Vpot)[-1]
                                                       self.input_density_wfn.V_potential())[-1]

    def get_esp4v0(self):
        # assert self.esp4v0 is None

        nthreads = psi4.get_num_threads()
        psi4.set_num_threads(1)
        print("ESP fitting starts. This might take a while.")
        x, y, z, _ = self.vp_basis.wfn.V_potential().get_np_xyzw()
        grid = np.array([x, y, z])
        grid = psi4.core.Matrix.from_array(grid.T)
        assert grid.shape[1] == 3, "Grid should be N*3 np.array"

        esp_calculator = psi4.core.ESPPropCalc(self.v0_wfn)
        self.esp4v0 = - esp_calculator.compute_esp_over_grid_in_memory(grid).np
        print("ESP fitting done")
        psi4.set_num_threads(nthreads)
        return

    def get_mol_vH(self):
        self.molecule.update_wfn_info()
        nthreads = psi4.get_num_threads()
        psi4.set_num_threads(1)
        print("ESP fitting starts. This might take a while.")

        x, y, z, _ = self.vp_basis.wfn.V_potential().get_np_xyzw()
        grid = np.array([x, y, z])
        grid = psi4.core.Matrix.from_array(grid.T)
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

    def Lagrangian_WuYang(self, v=None, fit4vxc=True):
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
            self.molecule.scf_inversion(1000, [Vks_a, Vks_b], add_vext=fit4vxc)
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

    def grad_WuYang(self, v=None, fit4vxc=True):
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

            self.molecule.scf_inversion(100, [Vks_a, Vks_b], add_vext=fit4vxc)
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

            self.molecule.scf_inversion(100, [Vks_a, Vks_b], add_vext=fit4vxc)
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

    def yang_L_curve_regularization(self, rgl_bs=np.e, rgl_epn=15, scipy_opt_method="trust-krylov", print_flag=False):
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
        print("Start L-curve search for regularization constant lambda. This might take a while..")
        for regularization_constant in rgl_list:
            print(regularization_constant)
            self.regularization_constant = regularization_constant
            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.scf_inversion(100, [Vks_a, Vks_b])
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
            self.molecule.scf_inversion(1000, [Vks_a, Vks_b])
            self.update_vout_constant()

            E_list.append(self.molecule.energy)
            L_list.append(v_result.fun)
            norm_list.append(self.regul_norm)
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

    def my_L_curve_regularization(self, rgl_bs=np.e, rgl_epn=15, starting_epn=1,
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
            self.molecule.scf_inversion(100, [Vks_a, Vks_b])
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
            nbf = int(v.shape[0]/2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]
            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
            self.molecule.scf_inversion(1000, [Vks_a, Vks_b])
            self.update_vout_constant()

            n_result = self.molecule.to_grid(self.molecule.Da.np + self.molecule.Db.np)
            P = np.linalg.norm(v_output_a) * np.sum(np.abs(n_result - n_input) * self.molecule.w)
            P_list.append(P)
            L_list.append(v_result.fun)
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

        self.molecule.scf_inversion(100, [Vks_a, Vks_b])
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
        self.molecule.scf_inversion(100, [Vks_a, Vks_b])
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
                              countinue_opt=False, find_vxc_grid=True):

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:

                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)
        if not countinue_opt:
            print("Zero the old result for a new calculation..")
            self.v_output = np.zeros_like(self.v_output)

            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.scf_inversion(100, [Vks_a, Vks_b])
            self.update_vout_constant()

        print("<<<<<<<<<<<<<<<<<<<<<<WuYang vxc Inversion %s<<<<<<<<<<<<<<<<<<<"%opt_method)

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("|n| before", np.sum(np.abs(dn)*self.molecule.w))
        if opt is None:
            opt = {
                "disp": False,
                "maxiter": maxiter,
                # "eps": 1e-7
                # "norm": 2,
                # "gtol": 1e-7
            }

        vp_array = optimizer.minimize(self.Lagrangian_WuYang, self.v_output,
                                      jac=self.grad_WuYang,
                                      hess=self.hess_WuYang,
                                      method=opt_method,
                                      options=opt,
                                      tol=tol)
        nbf = int(vp_array.x.shape[0] / 2)
        v_output_a = vp_array.x[:nbf]
        v_output_b = vp_array.x[nbf:]

        Vks_a = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        self.molecule.scf_inversion(100, [Vks_a, Vks_b])
        self.update_vout_constant()

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("Evaluation: ", vp_array.nfev)
        print("|jac|", np.linalg.norm(vp_array.jac), "|n|", np.sum(np.abs(dn)*self.molecule.w), "L after", vp_array.fun)
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
        self.v_output = vp_array.x

        if find_vxc_grid:
            self.get_vxc()
        return

    def find_vxc_manualNewton(self, maxiter=49, svd_rcond=None, c1=1e-4, c2=0.99, c3=1e-2,
                              svd_parameter=None, line_search_method="StrongWolfe",
                              BT_beta_threshold=1e-7, rho_conv_threshold=1e-3,
                              countinue_opt=False, find_vxc_grid=True):
        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        if not countinue_opt:
            print("Zero the old result for a new calculation..")
            self.v_output = np.zeros_like(self.v_output)
            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.scf_inversion(100, [Vks_a, Vks_b])
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
                svd = svd * 1.0001 / s[0]
                hess_inv = np.linalg.pinv(hess, rcond=svd)
                dv = -np.dot(hess_inv, jac)
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
                                                       amax=1)
                if beta is None:
                    LineSearch_converge_flag = True
                else:
                    self.v_output += beta * dv
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
                                                       amax=1)
                if beta is None:
                    LineSearch_converge_flag = True
                else:
                    self.v_output += beta * dv
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

        if find_vxc_grid:
            self.get_vxc()

        return hess, jac
# %% PDE constrained optimization on basis sets.
    def Lagrangian_constrainedoptimization(self, v=None):
        """
        Return Lagrange Multipliers from Nafziger and Jensen's constrained optimization.
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
            self.molecule.scf_inversion(1000, [Vks_a, Vks_b])
            self.update_vout_constant()

        dDa = self.molecule.Da.np - self.input_density_wfn.Da().np
        dDb = self.molecule.Db.np - self.input_density_wfn.Db().np
        dD = dDa + dDb

        g_uv = self.CO_weighted_cost()

        L = np.sum(dD * g_uv)

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

            self.molecule.scf_inversion(100, [Vks_a, Vks_b])
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

            self.molecule.scf_inversion(100, [Vks_a, Vks_b])
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

            self.molecule.scf_inversion(100, [Vks_a, Vks_b])
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
        self.molecule.scf_inversion(100, [Vks_a, Vks_b])
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
        self.molecule.scf_inversion(100, [Vks_a, Vks_b])
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
                                               countinue_opt=False, find_vxc_grid=True):

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:

                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        if not countinue_opt:
            print("Zero the old result for a new calculation..")
            self.v_output = np.zeros_like(self.v_output)

            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.scf_inversion(100, [Vks_a, Vks_b])
            self.update_vout_constant()

        print("<<<<<<<<<<<<<<<<<<<<<<Constrained Optimization vxc Inversion<<<<<<<<<<<<<<<<<<<")

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("|n| before", np.sum(np.abs(dn)*self.molecule.w))
        opt = {
            "disp": False,
            "maxiter": maxiter,
            # "eps": 1e-7
            # "norm": 2,
            "gtol": 1e-7
        }

        vp_array = optimizer.minimize(self.Lagrangian_constrainedoptimization,
                                      self.v_output,
                                      jac=self.grad_constrainedoptimization,
                                      method=opt_method,
                                      options=opt)

        nbf = int(vp_array.x.shape[0] / 2)
        v_output_a = vp_array.x[:nbf]
        v_output_b = vp_array.x[nbf:]

        Vks_a = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)

        self.molecule.scf_inversion(100, [Vks_a, Vks_b])
        self.update_vout_constant()

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("Evaluation", vp_array.nfev)
        print("|jac|", np.linalg.norm(vp_array.jac), "|n|", np.sum(np.abs(dn)*self.molecule.w), "L after", vp_array.fun)
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
        self.v_output = vp_array.x

        if find_vxc_grid:
            self.get_vxc()
        return vp_array

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
            self.molecule.scf_inversion(1000, [Vks_a, Vks_b])
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
            self.molecule.scf_inversion(1000, [Vks_a, Vks_b])
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
        self.molecule.scf_inversion(100, [Vks_a, Vks_b])
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
                                   countinue_opt=False,
                                   max_linesearch=100,
                                   find_vxc_grid=True):
        if not countinue_opt:
            print("Zero the old result for a new calculation..")
            self.v_output_grid = np.zeros_like(self.v_output_grid)
            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.scf_inversion(100, [Vks_a, Vks_b])
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
        self.molecule.scf_inversion(1000, [Vks_a, Vks_b])
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
                              countinue_opt=False, find_vxc_grid=True):
        if not countinue_opt:
            print("Zero the old result for a new calculation..")
            self.v_output_grid = np.zeros_like(self.v_output_grid)
            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.scf_inversion(100, [Vks_a, Vks_b])
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

        vp_array = optimizer.minimize(self.Lagrangian_WuYang_grid, self.v_output_grid,
                                      jac=self.grad_WuYang_grid,
                                      method=opt_method,
                                      options=opt)
        nbf = int(vp_array.x.shape[0] / 2)
        v_output_a = vp_array.x[:nbf]
        v_output_b = vp_array.x[nbf:]

        Vks_a = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        self.molecule.scf_inversion(1000, [Vks_a, Vks_b])
        self.update_vout_constant()

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("|jac|", np.linalg.norm(vp_array.jac), "|n|", np.sum(np.abs(dn)*self.molecule.w), "L after", vp_array.fun)
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
        self.v_output = vp_array.x

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
            self.molecule.scf_inversion(1000, [Vks_a, Vks_b])
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
            self.molecule.scf_inversion(1000, [Vks_a, Vks_b])
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
        self.molecule.scf_inversion(1000, [Vks_a, Vks_b])
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
                              countinue_opt=False, find_vxc_grid=True):

        if self.v0 != "HartreeLDAappext":
            print("Changing v0 to Hartree + LDA. Only works for LDA")
            self.get_HartreeLDA_v0()

        if self.four_overlap is None:
            self.four_overlap, _, _, _ = pdft.fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                          self.molecule.basis, self.molecule.mints)

        if not countinue_opt:
            print("Zero the old result for a new calculation..")
            if self.vext_app is None:
                self.vext_app = np.zeros(self.molecule.nbf**2)
            self.vext_app = np.zeros_like(self.vext_app)
            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.scf_inversion(100, [Vks_a, Vks_b], add_vext=False)
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

        vp_array = optimizer.minimize(self.Lagrangian_vext_WuYang_grid, self.vext_app,
                                      jac=self.grad_vext_WuYang_grid,
                                      method=opt_method,
                                      options=opt)
        nbf = int(vp_array.x.shape[0] / 2)
        v_output_a = vp_array.x[:nbf]
        v_output_b = vp_array.x[nbf:]

        Vks_a = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        self.molecule.scf_inversion(1000, [Vks_a, Vks_b], add_vext=False)
        self.update_vout_constant()

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("|jac|", np.linalg.norm(vp_array.jac), "|n|", np.sum(np.abs(dn)*self.molecule.w), "L after", vp_array.fun)
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
        self.v_output = vp_array.x

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
            self.molecule.scf_inversion(1000, [Vks_a, Vks_b], add_vext=False)
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
            self.molecule.scf_inversion(1000, [Vks_a, Vks_b], add_vext=False)
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
        self.molecule.scf_inversion(1000, [Vks_a, Vks_b], add_vext=False)
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
                              countinue_opt=False, find_vxc_grid=True):

        if self.v0 != "HartreeLDAappext":
            print("Changing v0 to Hartree + LDA. Only works for LDA")
            self.get_HartreeLDAappext_v0()

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)
        if not countinue_opt:
            print("Zero the old result for a new calculation..")
            self.v_output = np.zeros_like(self.v_output)

            Vks_a = psi4.core.Matrix.from_array(self.v0_Fock)
            Vks_b = psi4.core.Matrix.from_array(self.v0_Fock)
            self.molecule.scf_inversion(100, [Vks_a, Vks_b], add_vext=False)
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

        vp_array = optimizer.minimize(self.Lagrangian_WuYang, self.v_output,
                                      jac=self.grad_WuYang,
                                      hess=self.hess_WuYang,
                                      args=(False),
                                      method=opt_method,
                                      options=opt)
        nbf = int(vp_array.x.shape[0] / 2)
        v_output_a = vp_array.x[:nbf]
        v_output_b = vp_array.x[nbf:]

        Vks_a = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap,
                      v_output_a) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        Vks_b = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap,
                      v_output_b) + self.vout_constant * self.molecule.S.np + self.v0_Fock)
        self.molecule.scf_inversion(100, [Vks_a, Vks_b], add_vext=False)
        self.update_vout_constant()

        dDa = self.input_density_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_density_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa + dDb)
        print("|jac|", np.linalg.norm(vp_array.jac), "|n|", np.sum(np.abs(dn) * self.molecule.w), "L after",
              vp_array.fun)
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
            # nbf = int(vp_array.x.shape[0] / 2)
            # v_output_a = vp_array.x[:nbf]
            # v_output_b = vp_array.x[nbf:]
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

        self.molecule.scf_inversion(100, [Vks_a, Vks_b], add_vext=False)
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