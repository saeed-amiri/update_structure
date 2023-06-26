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

    update_aptes: pd.DataFrame  # Updated apted chains

    def __init__(self,
                 atoms: pd.DataFrame,  # All atoms coordinates
                 df_aptes: pd.DataFrame,  # All the APTES informations
                 h_positions: dict[int, np.ndarray]  # Index and coordinates
                 ) -> None:
        self.update_aptes = self.update_aptes_df(atoms, df_aptes, h_positions)

    def update_aptes_df(self,
                        atoms: pd.DataFrame,  # All the atoms info
                        df_aptes: pd.DataFrame,  # Aptes chains
                        h_positions: dict[int, np.ndarray]  # Pos for H
                        ) -> pd.DataFrame:
        """update the aptes dataframe by adding new HN3 atoms"""
        nh3_atoms: pd.DataFrame = self.__prepare_hydrogens(atoms, h_positions)
        return self.__append_hydrogens(df_aptes, nh3_atoms)

    @staticmethod
    def __append_hydrogens(df_aptes: pd.DataFrame,  # Aptes chains
                           hn3_atoms: pd.DataFrame  # H atoms to append
                           ) -> pd.DataFrame:
        """append NH3 atoms to the main df"""
        return pd.concat([df_aptes, hn3_atoms], ignore_index=False)

    @staticmethod
    def __prepare_hydrogens(atoms: pd.DataFrame,  # All the atoms info
                            h_positions: dict[int, np.ndarray]  # Pos for H
                            ) -> pd.DataFrame:
        """prepare the aptes based on the structure of the main df"""
        final_atom: int = atoms.iloc[-1]['atom_id']
        cols: list[str] = ['records', 'atom_id', 'atom_name',
                           'residue_number', 'residue_name', 'x', 'y', 'z',
                           'occupancy', 'temperature', 'atom_symbol']
        hn3_atoms: pd.DataFrame = pd.DataFrame(columns=cols)

        for i, (key, value) in enumerate(h_positions.items()):
            hn3_atoms.loc[i] = \
                ['ATOM', final_atom + i + 1, 'HN3', key, 'APT', value[0],
                 value[1], value[2], 1.0, 0.0, 'H']

        return hn3_atoms


if __name__ == '__main__':
    data = ionization.IonizationSol(sys.argv[1])
    updated = UpdateAptesDf(data.atoms,
                            data.residues_atoms['APT'],
                            data.h_porotonations)
