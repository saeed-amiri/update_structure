"""In order to incorporate counterions into the system, the script
searches for unoccupied spaces within the water section and identifies
all present atoms. It ensures that the placement of counterions does
not overlap with any existing atoms based on the number of new
protonation."""


import sys
import numpy as np
import pandas as pd
import protonating as proton


class IonizationSol(proton.FindHPosition):
    """ionizing the water phase of the box. The number is equal to the
    number of deprotonation of the APTES"""
    def __init__(self,
                 fname: str  # Name of the pdb file
                 ) -> None:
        super().__init__(fname)
        self.mk_ionization()

    def mk_ionization(self) -> None:
        """get the numbers of ions and their locations in the water
        phase"""
        x_dims: np.ndarray  # Dimensions of the sol box in x
        y_dims: np.ndarray  # Dimensions of the sol box in y
        z_dims: np.ndarray  # Dimensions of the sol box in z
        sol_atoms: pd.DataFrame = self.__get_sol_phase_atoms()
        x_dims, y_dims, z_dims = self.__get_box_size(sol_atoms)

    @staticmethod
    def __get_box_size(sol_atoms: pd.DataFrame  # All the atoms below NP
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """get the dimension of the sol box (water and ions)"""
        x_dims = np.array([np.min(sol_atoms['x']), np.max(sol_atoms['x'])])
        y_dims = np.array([np.min(sol_atoms['y']), np.max(sol_atoms['y'])])
        z_dims = np.array([np.min(sol_atoms['z']), np.max(sol_atoms['z'])])
        return x_dims, y_dims, z_dims

    def __get_sol_phase_atoms(self) -> pd.DataFrame:
        """get all the atom below interface, in the water section"""
        tresh_hold = float(self.param['INTERFACE']+100-self.np_diameter)
        return self.atoms[self.atoms['z'] < tresh_hold]


if __name__ == '__main__':
    ioning = IonizationSol(sys.argv[1])
