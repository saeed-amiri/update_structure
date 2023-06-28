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
import typing
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

    atoms_updated: pd.DataFrame  # Updated atoms section of the itp file

    def __init__(self,
                 atoms_np: pd.DataFrame,  # Atoms form the itp file for NP
                 hn3: pd.DataFrame  # New HN3 to add to the atoms
                 ) -> None:
        self.atoms_updated = self.update_atoms(atoms_np, hn3)

    def update_atoms(self,
                     atoms_np: pd.DataFrame,  # Atoms form the itp file for NP
                     hn3: pd.DataFrame  # New HN3 to add to the atoms
                     ) -> pd.DataFrame:
        """update the atoms"""
        # Sanity check of indeces and return the atom index in atoms
        lst_atom: np.int64  # index of the final atoms
        lst_atom = self.__get_indices(atoms_np, hn3)
        # Get only APT atoms
        atoms: pd.DataFrame = atoms_np[atoms_np['resname'] == 'APT']
        # Get COR atoms
        cor_atoms: pd.DataFrame = atoms_np[atoms_np['resname'] == 'COR']
        # Get information for protonated H-N group from the itp file
        h_n_df: pd.DataFrame = self.__get_n_h_proton_info(atoms)
        # Make a dataframe in format of itp for new nh3
        prepare_hn3: pd.DataFrame = self.__mk_hn3_itp_df(hn3, h_n_df, lst_atom)
        # Update N, HN1, and HN2 charges in the protonated chains
        atoms = \
            self.__update_chains(atoms, h_n_df, list(hn3['residue_number']))
        updated_aptes: pd.DataFrame = self.__concat_aptes(prepare_hn3, atoms)
        # get the final atoms section of itp file
        updates_np: pd.DataFrame = self.mk_np(cor_atoms, updated_aptes)
        return updates_np

    @staticmethod
    def mk_np(cor_atoms: pd.DataFrame,  # Atoms belong to sili
              updated_aptes: pd.DataFrame  # Chains with new hn3 and updated q
              ) -> pd.DataFrame:
        """make the final dataframe by appending updated aptes and core
        atoms of the nanoparticle"""
        return pd.concat([cor_atoms, updated_aptes],
                         axis=0, ignore_index=False)

    @staticmethod
    def __concat_aptes(prepare_hn3: pd.DataFrame,  # Prepared HN3 dataframe
                       atoms: pd.DataFrame  # Updated chain with charges
                       ) -> pd.DataFrame:
        """concate the dataframes and sort them based on the residue
        indices"""
        updated_atoms: pd.DataFrame = \
            pd.concat([atoms, prepare_hn3], axis=0, ignore_index=False)

        updated_atoms = updated_atoms.sort_values(by=['resnr', 'atomnr'],
                                                  ascending=[True, True])
        return updated_atoms

    @staticmethod
    def __update_chains(atoms: pd.DataFrame,  # APTES atoms
                        h_n_df: pd.DataFrame,  # Protonated N-H group info
                        res_numbers: list[int]  # Index of the chains to prtons
                        ) -> pd.DataFrame:
        """update the N, HN1, and HN2 in the chains which should be
        updated"""
        n_q: float  # Charge of N atom in protonated state
        h1_q: float  # Charge of HN1 atom in protonated state
        h2_q: float  # Charge of HN1 atom in protonated state
        n_q = UpdateAtom.__get_info_dict(h_n_df, 'N')[1]['charge']
        h1_q = UpdateAtom.__get_info_dict(h_n_df, 'HN1')[1]['charge']
        h2_q = UpdateAtom.__get_info_dict(h_n_df, 'HN2')[1]['charge']
        # Create a condition for selecting rows that need to be updated
        condition = (atoms['atomname'].isin(['N', 'HN1', 'HN2'])) & \
                    (atoms['resnr'].isin(res_numbers))

        # Update the 'charge' column for the selected rows
        atoms.loc[condition, 'charge'] = \
            atoms.loc[condition, 'atomname'].map({'N': n_q,
                                                  'HN1': h1_q,
                                                  'HN2': h2_q})
        return atoms

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

    @staticmethod
    def __mk_hn3_itp_df(hn3: pd.DataFrame,  # New HN3 atoms, in pdb format
                        h_n_df: pd.DataFrame,  # Info for H-N protonated group
                        lst_atom: np.int64  # Last atom index in hn3
                        ) -> pd.DataFrame:
        """make a dataframe in the itp format, for appending to the
        main atom section"""
        columns: list[str]  # Name of the columns
        info: dict[str, typing.Any]  # Info in the row of the df
        columns, info = UpdateAtom.__get_info_dict(h_n_df, 'HN3')
        hn3_itp = pd.DataFrame(columns=columns)
        for item, row in hn3.iterrows():
            atom_id = lst_atom + item + 1
            hn3_itp.loc[item] = [atom_id,
                                 info['atomtype'],
                                 int(row['residue_number']),
                                 info['resname'],
                                 info['atomname'],
                                 info['chargegrp'],
                                 info['charge'],
                                 info['mass'],
                                 info['element']]
        return hn3_itp

    @staticmethod
    def __get_info_dict(df_info: pd.DataFrame,  # The data frame to take info
                        key: str,  # The name of the atom
                        ) -> tuple[list[str], dict[str, typing.Any]]:
        """return dictionay of info in the row of df"""
        columns: list[str] = ['atomnr', 'atomtype', 'resnr', 'resname',
                              'atomname', 'chargegrp', 'charge', 'mass',
                              'element']
        info: dict[str, typing.Any] = {}
        for item in columns:
            info[item] = \
                df_info.loc[df_info['atomname'] == key, item].values[0]
        return columns, info

    @staticmethod
    def __get_indices(atoms: pd.DataFrame,  # Atoms of the itp file
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
            sys.exit(f'{bcolors.FAIL}{UpdateAtom.__module__}:\n'
                     '\tThere is mismatch in the new HN3 and initial'
                     f' APTES list.\n{bcolors.ENDC}')
        return np.max(np.array([max_atomnr, lst_atomnr]))


if __name__ == '__main__':
    print("This script should be call from 'updata_pdb_itp.py")
