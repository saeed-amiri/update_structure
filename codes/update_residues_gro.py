"""To properly format GRO files, velocities columns must be included
in the input file. It is essential to index the atoms and residues
correctly. The indexing of the atoms and residues does not start at
zero; the index of the first residue is actually one greater than the
last residue written in the input file. Additionally, IONs will
increase the number of residues and atom numbers, but the new HN3 only
increases the number of atoms.
All the dataframes are needed. Everthing should be updated.
"""

import sys
import numpy as np
import pandas as pd
import ionization
import logger


class UpdateAptesDf:
    """
    Updates APTES dataframe, by adding the hydrogens. Based on the
    positions, we have to make H atoms to be added to the gro file.
    """

    info_msg: str = 'Message:\n'  # Message to pass for logging and writing

    def __init__(self,
                 df_aptes: pd.DataFrame,  # All the APTES informations
                 h_positions: dict[int, np.ndarray],  # Index and coordinates
                 h_velocities: dict[int, np.ndarray]  # Index and velocities
                 ) -> None:
        self.update_aptes, self.new_nh3 = \
            self.update_aptes_df(df_aptes, h_positions, h_velocities)

    def update_aptes_df(self,
                        df_aptes: pd.DataFrame,  # Aptes chains
                        h_positions: dict[int, np.ndarray],  # H positions
                        h_velocities: dict[int, np.ndarray]  # H velocities
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """update the aptes dataframe by adding new HN3 atoms"""
        nh3_atoms: pd.DataFrame = \
            self.prepare_hydrogens(h_positions, h_velocities)
        all_aptes: pd.DataFrame = self.__append_hydrogens(df_aptes, nh3_atoms)
        updated_aptes: pd.DataFrame = self.__update_aptes_atom_id(all_aptes)
        return updated_aptes, nh3_atoms

    @staticmethod
    def __update_aptes_atom_id(df_aptes: pd.DataFrame  # APTES from file
                               ) -> pd.DataFrame:
        """reindex atoms and residues ids from 1"""
        df_c: pd.DataFrame = df_aptes.copy()
        atom_ind: list[int] = [i+1 for i in range(len(df_c))]
        df_c['atom_id'] = atom_ind
        return df_c

    @staticmethod
    def __append_hydrogens(df_aptes: pd.DataFrame,  # Aptes chains
                           hn3_atoms: pd.DataFrame  # H atoms to append
                           ) -> pd.DataFrame:
        """append NH3 atoms to the main df"""
        return pd.concat([df_aptes, hn3_atoms], ignore_index=False)

    @staticmethod
    def prepare_hydrogens(h_positions: dict[int, np.ndarray],  # H positions
                          h_velocities: dict[int, np.ndarray]  # H velocities
                          ) -> pd.DataFrame:
        """prepare the aptes based on the structure of the main df"""
        cols: list[str] = \
            ['residue_number', 'residue_name', 'atom_name', 'atom_id',
             'x', 'y', 'z', 'vx', 'vy', 'vz']
        hn3_atoms: pd.DataFrame = pd.DataFrame(columns=cols)

        for i, (key, pos) in enumerate(h_positions.items()):
            atom_id = i+1
            velo = h_velocities[key]
            hn3_atoms.loc[i] = \
                [key, 'APT', 'HN3', atom_id, pos[0],
                 pos[1], pos[2], velo[0], velo[1], velo[2]]
        return hn3_atoms


# Helper function to update index in pdb fasion
def get_pdb_index(ind: int,  # Index which should be updated
                  final_atom: int,  # Last index of atom in the system
                  pdb_max: int = 99999  # If not atom change it
                  ) -> int:
    """updata index (for atoms or residues). In pdb file, atom index
    cannot be bigger then 99999 and residues 9999. Afterwards,
    it started from zero"""
    new_ind: int = final_atom + ind
    if new_ind > pdb_max:
        new_ind -= pdb_max
    return new_ind


class UpdateResidues:
    """get all the dataframes as an object"""

    # To append all the residues to the dict
    updated_residues: dict[str, pd.DataFrame] = {}
    # To append new atoms to the main atoms
    updated_atoms: pd.DataFrame
    # To return new atoms seperatly
    new_hn3: pd.DataFrame
    new_ions: pd.DataFrame

    def __init__(self,
                 fname: str  # Name of the input file (pdb)
                 ) -> None:
        data = ionization.IonizationSol(fname)
        self.get_residues(data)

    def get_residues(self,
                     data: ionization.IonizationSol  # All the data
                     ) -> None:
        """get all the residues"""
        """get all the residues"""
        self.updated_residues['SOL'] = self.get_sol(data)
        self.updated_residues['D10'] = self.get_oil(data)
        self.updated_residues['COR'] = self.get_cor(data)
        # self.updated_residues['APT'], new_hn3 =
        self.get_aptes(data)

    @staticmethod
    def get_atoms(atoms: pd.DataFrame,  # Initial system
                  new_hn3: pd.DataFrame,  # New NH3 atoms
                  new_ions: pd.DataFrame,  # New Ions atoms
                  ) -> pd.DataFrame:
        """append the new atoms to the main dataframe with all atoms"""
        return pd.concat([atoms, new_hn3, new_ions])

    @staticmethod
    def get_aptes(data: ionization.IonizationSol  # All the data
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """get updated aptes dataframe"""
        updated_aptes = UpdateAptesDf(data.residues_atoms['APT'],
                                      data.h_porotonations,
                                      data.h_velocities)
        # return updated_aptes.update_aptes, updated_aptes.new_nh3

    @staticmethod
    def get_sol(data: ionization.IonizationSol  # All the data
                ) -> pd.DataFrame:
        """return water residues"""
        return data.residues_atoms['SOL']

    @staticmethod
    def get_oil(data: ionization.IonizationSol  # All the data
                ) -> pd.DataFrame:
        """return oil residues"""
        return data.residues_atoms['D10']

    @staticmethod
    def get_cor(data: ionization.IonizationSol  # All the data
                ) -> pd.DataFrame:
        """return core atoms of NP residues"""
        return data.residues_atoms['COR']


if __name__ == '__main__':
    UpdateResidues(sys.argv[1])
