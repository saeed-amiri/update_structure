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
                 atoms_np: pd.DataFrame,  # Atoms form the itp file for NP
                 hn3: pd.DataFrame  # New HN3 to add to the atoms
                 ) -> None:
        self.update_atoms(atoms_np, hn3)

    def update_atoms(self,
                     atoms_np: pd.DataFrame,  # Atoms form the itp file for NP
                     hn3: pd.DataFrame  # New HN3 to add to the atoms
                     ) -> None:
        """update the atoms"""
        # Sanity check of indeces and return the atom index in atoms
        lst_atom: np.int64  # index of the final atoms
        lst_atom = self.__get_indices(atoms_np, hn3)
        # Get only APT atoms
        atoms: pd.DataFrame = atoms_np[atoms_np['resname'] == 'APT']
        # Get information for HN3 from the itp file itself
        self.__get_n_h_proton_info(atoms)

    @staticmethod
    def __get_n_h_proton_info(atoms: pd.DataFrame,  # Atoms of the itp file
                              ) -> pd.DataFrame:
        """get and return info for protonated H-N group from the itp
        file"""
        df_tmp: pd.DataFrame  # One protonated APTES
        df_tmp = atoms[atoms['atomname'] == 'HN3']
        protonated_apt: list[int]  # Indices of alreay protonated APTES
        protonated_apt = list(atoms['resnr'])
        # Just an id!
        rand_id: int = protonated_apt[0]
        df_tmp = atoms[atoms['resnr'] == rand_id]
        df_one: pd.DataFrame = \
            df_tmp[df_tmp['atomname'].isin(['N', 'HN1', 'HN2', 'HN3'])]
        if atoms[atoms['atomname'] == 'HN3'].empty:
            sys.exit(f'{bcolors.FAIL}{UpdateAtom.__module__}: \n'
                     '\tError! There is no HN3 in the chosen protonated'
                     f'branch\n{bcolors.ENDC}')
        return df_one

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
