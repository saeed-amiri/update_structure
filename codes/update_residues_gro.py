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

    update_aptes: pd.DataFrame  # Updated APTES df
    new_nh3: pd.DataFrame  # All the new HN3 atoms

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


class UpdateSolDf:
    """preparing water (SOL) residue for updating. Nothing needs to be
    change"""

    update_waters: pd.DataFrame  # Updated APTES df

    def __init__(self,
                 atoms: pd.DataFrame  # All SOL atoms
                 ) -> None:
        self.update_waters = self.update_water_df(atoms)
        self.update_waters.to_csv('water_test.gro', sep=' ')

    @staticmethod
    def update_water_df(atoms: pd.DataFrame  # All SOL atoms
                        ) -> pd.DataFrame:
        """update water atoms if needed to be updated"""
        return atoms


class UpdateCorDf:
    """preparing COR residue for updating. Nothing needs to be
    change"""

    update_cor: pd.DataFrame  # Updated COR df

    def __init__(self,
                 atoms: pd.DataFrame  # All COR atoms
                 ) -> None:
        self.update_cor = self.update_cor_df(atoms)
        self.update_cor.to_csv('cor_test.gro', sep=' ')

    @staticmethod
    def update_cor_df(atoms: pd.DataFrame  # All COR atoms
                      ) -> pd.DataFrame:
        """update core (COR) atoms if needed to be updated"""
        atom_ids: list[int] = [i+1 for i in atoms.index]
        atoms['atom_id'] = atom_ids
        return atoms


class UpdateOdaDf:
    """preparing ODA residue for updating.The residue index should be
    changed"""

    update_oda: pd.DataFrame  # Updated ODA df

    def __init__(self,
                 atoms: pd.DataFrame,  # All ODA atoms
                 protonation_nr: int  # Number of protonation
                 ) -> None:
        self.update_oda = self.update_oda_df(atoms, protonation_nr)
        self.update_oda.to_csv('oda_test.gro', sep=' ')

    @staticmethod
    def update_oda_df(atoms: pd.DataFrame,  # All ODA atoms
                      protonation_nr: int  # Number of protonation
                      ) -> pd.DataFrame:
        """update ODA atoms if needed to be updated"""
        df_c: pd.DataFrame = atoms.copy()
        updated_res: list[int] = atoms['residue_number'] + protonation_nr
        atoms_id: list[int] = [i+1 for i in atoms.index]
        df_c['residue_number'] = updated_res
        df_c['atoms_id'] = atoms_id
        return df_c


class UpdateIonDf:
    """preparing ION residue for updating. The new counterions should
    be added to the initial ones"""

    update_ion: pd.DataFrame  # Updated ION df

    def __init__(self,
                 atoms: pd.DataFrame,  # All ION atoms
                 ion_poses: list[np.ndarray],  # Position for the new ions
                 ion_velos: pd.DataFrame  # Velocities for the new ions
                 ) -> None:
        self.update_ion = self.update_ion_df(atoms, ion_poses, ion_velos)
        self.update_ion.to_csv('ion_test.gro', sep=' ')

    def update_ion_df(self,
                      atoms: pd.DataFrame,  # All ION atoms
                      ion_poses: list[np.ndarray],  # Position for the new ions
                      ion_velos: pd.DataFrame  # Velocities for the new ions
                      ) -> pd.DataFrame:
        """update ions atoms if needed to be updated"""
        final_res: int = atoms['residue_number'].iloc[-1]
        final_atom: int = atoms['atom_id'].iloc[-1]
        new_ions: pd.DataFrame = \
            self.prepare_ions(ion_poses, ion_velos, final_res, final_atom)
        updated_ions: pd.DataFrame = self.__append_ions(atoms, new_ions)
        updated_ions.to_csv('ions_test.gro', sep=' ')
        return atoms

    @staticmethod
    def prepare_ions(ion_poses: dict[int, np.ndarray],  # Ions positions
                     ion_velos: dict[int, np.ndarray],  # Ions velocities
                     final_res: int,  # Last residue in the CLA df
                     final_atom: int  # Last atom in the CLA df
                     ) -> pd.DataFrame:
        """prepare the IONS based on the structure of the main df"""
        cols: list[str] = \
            ['residue_number', 'residue_name', 'atom_name', 'atom_id',
             'x', 'y', 'z', 'vx', 'vy', 'vz']
        new_ions: pd.DataFrame = pd.DataFrame(columns=cols)
        for i, (pos, vel) in enumerate(zip(ion_poses, ion_velos)):
            res_id: int = final_res + i + 1
            atom_id: int = final_atom + i + 1
            new_ions.loc[i] = [res_id, 'CLA', 'CLA', atom_id, pos[0],
                               pos[1], pos[2], vel[0], vel[1], vel[2]]
        return new_ions

    @staticmethod
    def __append_ions(ions: pd.DataFrame,  # initial ions
                      new_ions: pd.DataFrame  # New ions atoms to append
                      ) -> pd.DataFrame:
        """append new ions atoms to the main df"""
        return pd.concat([ions, new_ions], ignore_index=False)


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
        self.updated_residues['SOL'] = self.get_sol(data)
        self.updated_residues['D10'] = self.get_oil(data)
        self.updated_residues['COR'] = self.get_cor(data)
        # self.updated_residues['APT'], new_hn3 =
        self.get_aptes(data)
        self.get_ions(data)
        self.get_oda(data)

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

    @staticmethod
    def get_ions(data: ionization.IonizationSol  # All the data
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """get updated ions data frame"""
        updated_ions = UpdateIonDf(data.residues_atoms['CLA'],
                                   data.ion_poses,
                                   data.ion_velos)
        # return updated_ions.update_ions, updated_ions.new_ions

    @staticmethod
    def get_oda(data: ionization.IonizationSol  # All the data
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """get updated ions data frame"""
        updated_oda = UpdateOdaDf(data.residues_atoms['ODN'],
                                  int(len(data.ion_poses)))
        # return updated_ions.update_ions, updated_ions.new_ions


if __name__ == '__main__':
    UpdateResidues(sys.argv[1])
