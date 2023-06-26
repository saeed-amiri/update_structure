"""It is necessary to update the names and charges of APTES chains
when their protonation changes to read the correct parameters from the
Forcefield data files. The changes that need to be made are as
follows:

"""

import sys
import numpy as np
import pandas as pd
import ionization


class UpdateAptesDf:
    """Updates APTES dataframe, by adding the hydrogens. Based on the
    positions, we have to make H atoms to be added to the pdb file"""

    update_aptes: pd.DataFrame  # Updated aptes chains
    new_nh3: pd.DataFrame  # New NH3 atoms

    def __init__(self,
                 atoms: pd.DataFrame,  # All atoms coordinates
                 df_aptes: pd.DataFrame,  # All the APTES informations
                 h_positions: dict[int, np.ndarray]  # Index and coordinates
                 ) -> None:
        self.update_aptes, self.new_nh3 = \
            self.update_aptes_df(atoms, df_aptes, h_positions)

    def update_aptes_df(self,
                        atoms: pd.DataFrame,  # All the atoms info
                        df_aptes: pd.DataFrame,  # Aptes chains
                        h_positions: dict[int, np.ndarray]  # Pos for H
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """update the aptes dataframe by adding new HN3 atoms"""
        nh3_atoms: pd.DataFrame = self.prepare_hydrogens(atoms, h_positions)
        return self.__append_hydrogens(df_aptes, nh3_atoms), nh3_atoms

    @staticmethod
    def __append_hydrogens(df_aptes: pd.DataFrame,  # Aptes chains
                           hn3_atoms: pd.DataFrame  # H atoms to append
                           ) -> pd.DataFrame:
        """append NH3 atoms to the main df"""
        return pd.concat([df_aptes, hn3_atoms], ignore_index=False)

    @staticmethod
    def prepare_hydrogens(atoms: pd.DataFrame,  # All the atoms info
                          h_positions: dict[int, np.ndarray]  # Pos for H
                          ) -> pd.DataFrame:
        """prepare the aptes based on the structure of the main df"""
        final_atom: int = atoms.iloc[-1]['atom_id']
        cols: list[str] = ['records', 'atom_id', 'atom_name',
                           'residue_number', 'residue_name', 'x', 'y', 'z',
                           'occupancy', 'temperature', 'atom_symbol']
        hn3_atoms: pd.DataFrame = pd.DataFrame(columns=cols)

        for i, (key, value) in enumerate(h_positions.items()):
            atom_id = get_pdb_index(i+1, final_atom)
            hn3_atoms.loc[i] = \
                ['ATOM', atom_id, 'HN3', key, 'APT', value[0],
                 value[1], value[2], 1.0, 0.0, 'H']

        return hn3_atoms


class UpdateIonDf:
    """update ion dataframe by adding the prepared ions to it"""

    update_ions: pd.DataFrame  # Updated ions
    new_ions: pd.DataFrame  # New ions

    def __init__(self,
                 atoms: pd.DataFrame,  # All the atoms
                 ions_df: pd.DataFrame,  # Initial ions df
                 ions_poses: np.ndarray  # Postions of the ions
                 ) -> None:
        self.update_ions, self.new_ions = \
            self.update_ion_df(atoms, ions_df, ions_poses)

    def update_ion_df(self,
                      atoms: pd.DataFrame,  # All the atoms
                      ions_df: pd.DataFrame,  # Initial ions df
                      ions_poses: np.ndarray  # Postions of the ions
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """prepare and update the ions dataframe"""
        ions_atoms: pd.DataFrame = self.prepare_ions(atoms, ions_poses)
        return self.__append_ions(ions_df, ions_atoms), ions_atoms

    @staticmethod
    def __append_ions(ions_df: pd.DataFrame,  # Initial datafrme of ions
                      ions_atoms: pd.DataFrame  # New ions atoms
                      ) -> pd.DataFrame:
        """append the new ions to the old one"""
        return pd.concat([ions_df, ions_atoms])

    @staticmethod
    def prepare_ions(atoms: pd.DataFrame,  # All the atoms
                     ions_poses: np.ndarray  # Ions poistions
                     ) -> None:
        """prepare dataframe for ions"""
        # Get the final index of the atoms after adding NH3 to the system
        final_atom: int = atoms.iloc[-1]['atom_id'] + len(ions_poses)
        # Get final index of the residues to add ions residues
        final_res: int = atoms.iloc[-1]['residue_number']
        cols: list[str] = ['records', 'atom_id', 'atom_name',
                           'residue_number', 'residue_name', 'x', 'y', 'z',
                           'occupancy', 'temperature', 'atom_symbol']
        ions_atoms: pd.DataFrame = pd.DataFrame(columns=cols)
        for i, pos in enumerate(ions_poses):
            atom_id = get_pdb_index(i+1, final_atom)
            res_id = get_pdb_index(i+1, final_res, pdb_max=9999)
            ions_atoms.loc[i] = \
                ['ATOM', atom_id, 'CLA', res_id, 'CLA', pos[0],
                 pos[1], pos[2], 1.0, 0.0, 'CL']
        return ions_atoms


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

    def __init__(self,
                 fname: str  # Name of the input file (pdb)
                 ) -> None:
        data = ionization.IonizationSol(fname)
        self.get_residues(data)

    def get_residues(self,
                     data: ionization.IonizationSol  # All the data
                     ) -> None:
        """get all the residues"""
        self.updated_residues['APT'] = self.get_aptes(data)
        self.updated_residues['SOL'] = self.get_sol(data)
        self.updated_residues['D10'] = self.get_oil(data)
        self.updated_residues['COR'] = self.get_cor(data)
        self.updated_residues['CLA'] = self.get_ions(data)

    @staticmethod
    def get_ions(data: ionization.IonizationSol  # All the data
                 ) -> pd.DataFrame:
        """get updated ions data frame"""
        updated_ions = UpdateIonDf(data.atoms,
                                   data.residues_atoms['CLA'],
                                   data.ion_poses)
        return updated_ions.update_ions

    @staticmethod
    def get_aptes(data: ionization.IonizationSol  # All the data
                  ) -> pd.DataFrame:
        """get updated aptes dataframe"""
        updated_aptes = UpdateAptesDf(data.atoms,
                                      data.residues_atoms['APT'],
                                      data.h_porotonations)
        return updated_aptes.update_aptes

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
