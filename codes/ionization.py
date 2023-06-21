"""In order to incorporate counterions into the system, the script
searches for unoccupied spaces within the water section and identifies
all present atoms. It ensures that the placement of counterions does
not overlap with any existing atoms based on the number of new
protonation."""


import sys
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
        sol_atoms: pd.DataFrame = self.__get_sol_phase_atoms()

    def __get_sol_phase_atoms(self) -> pd.DataFrame:
        """get all the atom below interface, in the water section"""
        tresh_hold: float = \
            self.param['INTERFACE']+100-self.param['INTERFACE_WIDTH']/2
        return self.atoms[self.atoms['z'] < tresh_hold]


if __name__ == '__main__':
    ioning = IonizationSol(sys.argv[1])
