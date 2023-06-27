"""The script updates the "APT_COR.itp" file using the HN3 atoms set by
update_residue.py. To do this, we need to add the correct information
for the name and charge for the new HN3 in the atoms section.
Additionally, for chains that have been protonated, It also updates
the name and charges for N, HN1, and HN2 to match the new values.
The script needs to add bonds between N and the new HN3. Since force-
field values are set in different files, we only need to update the
sections for the extra H in each chain, including angles and dihedrals.
"""

import sys
import numpy as np
import pandas as pd
import itp_to_df as itp
from colors_text import TextColor as bcolors


class UpdateItp(itp.Itp):
    """updating the itp file for each section"""
    # object from itp class
    # atoms: pd.DataFrame -> Section of initial itp file
    # bonds: pd.DataFrame -> Section of initial itp file
    # angles: pd.DataFrame -> Section of initial itp file
    # dihedrals: pd.DataFrame -> Section of initial itp file
    # molecules: pd.DataFrame -> Section of initial itp file

    def __init__(self,
                 fname: str,  # Name of the itp file
                 hn3: pd.DataFrame  # Information for new NH3 atoms
                 ) -> None:
        super().__init__(fname)
        self.update_itp(hn3)

    def update_itp(self,
                   hn3: pd.DataFrame  # NH3 atoms
                   ) -> None:
        """get all the data and return final df"""
        UpdateAtom(self.atoms, hn3)


class UpdateAtom:
    """update atom section by adding new hn3 and updating the N, HN1,
    HN2"""
    def __init__(self,
                 atoms: pd.DataFrame,  # Atoms form the itp file
                 hn3: pd.DataFrame  # New HN3 to add to the atoms
                 ) -> None:
        self.update_atoms(atoms, hn3)

    def update_atoms(self,
                     atoms: pd.DataFrame,  # Atoms form the itp file
                     hn3: pd.DataFrame  # New HN3 to add to the atoms
                     ) -> None:
        """update the atoms"""
        lst_atom: np.int64  # index of the final atoms
        lst_atom = self.__get_indices(atoms, hn3)

    def __get_indices(self,
                      atoms: pd.DataFrame,  # Atoms of the itp file
                      hn3: pd.DataFrame  # New HN3 to add to the atoms
                      ) -> np.int64:
        """return the maximum value of the atoms and check mismatch
        residue numbers"""
        atoms['atomnr'] = pd.to_numeric(atoms['atomnr'], errors='coerce')
        atoms['resnr'] = pd.to_numeric(atoms['resnr'], errors='coerce')

        max_atomnr: np.int64 = np.max(atoms['atomnr'])
        lst_atomnr: int = atoms['atomnr'].iloc[-1]
        max_resnr: np.int64 = np.max(atoms['resnr'])
        lst_resnr: int = atoms['resnr'].iloc[-1]

        lst_nh3_res: np.int64  # index of the final residue
        lst_nh3_res = list(hn3['residue_number'])[-1]
        if np.max(np.array([max_resnr, lst_resnr])) != lst_nh3_res:
            sys.exit(f'{bcolors.FAIL}{self.__module__}:\n'
                     '\tThere is mismatch in the new HN3 and initial'
                     f' APTES list.\n{bcolors.ENDC}')
        return np.max(np.array([max_atomnr, lst_atomnr]))


if __name__ == '__main__':
    print("This script should be call from 'updata_pdb_itp.py")
