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


class UpdateAptesDf:
    """
    Updates APTES dataframe, by adding the hydrogens. Based on the
    positions, we have to make H atoms to be added to the gro file.
    """

    update_aptes:  dict[str, pd.DataFrame]  # Updated APTES df
    new_nh3:  dict[str, pd.DataFrame]  # All the new HN3 atoms

    def __init__(self,
                 df_aptes: dict[str, pd.DataFrame],  # All the APTES informatio
                 h_positions: dict[str, dict[int, np.ndarray]],  # Coordinates
                 h_velocities: dict[str, dict[int, np.ndarray]]  # Velocities
                 ) -> None:
        self.update_aptes, self.new_nh3 = \
            self.update_aptes_df(df_aptes, h_positions, h_velocities)

    def update_aptes_df(self,
                        df_aptes: dict[str, pd.DataFrame],  # Aptes chains
                        h_positions: dict[str, dict[int, np.ndarray]],  # H pos
                        h_velocities: dict[str, dict[int, np.ndarray]]  # H vel
                        ) -> tuple[dict[str, pd.DataFrame],
                                   dict[str, pd.DataFrame]]:
        """update the aptes dataframe by adding new HN3 atoms"""
        nh3_atoms: dict[str, pd.DataFrame] = \
            self.prepare_hydrogens(h_positions, h_velocities)
        all_aptes:  dict[str, pd.DataFrame] = \
            self.__append_hydrogens(df_aptes, nh3_atoms)
        updated_aptes:  dict[str, pd.DataFrame] = \
            self.__update_aptes_atom_id(all_aptes)
        return updated_aptes, nh3_atoms

    @staticmethod
    def __update_aptes_atom_id(df_aptes: dict[str, pd.DataFrame]  # APTES
                               ) -> dict[str, pd.DataFrame]:
        """reindex atoms and residues ids from 1"""
        id_updated_aptes: dict[str, pd.DataFrame] = {}
        for aptes, item in df_aptes.items():
            df_c: pd.DataFrame = item.copy()
            atom_ind: list[int] = [i+1 for i in range(len(df_c))]
            df_c['atom_id'] = atom_ind
            id_updated_aptes[aptes] = df_c
            del df_c
        return id_updated_aptes

    @staticmethod
    def __append_hydrogens(df_aptes: dict[str, pd.DataFrame],  # Aptes chains
                           hn3_atoms: pd.DataFrame  # H atoms to append
                           ) -> dict[str, pd.DataFrame]:
        """append NH3 atoms to the main df"""
        appneded_hn3_df: dict[str, pd.DataFrame] = {}
        for aptes, item in df_aptes.items():
            appneded_hn3_df[aptes] = \
                pd.concat([item, hn3_atoms[aptes]], ignore_index=False)
        return appneded_hn3_df

    @staticmethod
    def prepare_hydrogens(h_positions: dict[str, dict[int, np.ndarray]],
                          h_velocities: dict[str, dict[int, np.ndarray]]
                          ) -> dict[str, pd.DataFrame]:
        """
        Prepare the aptes based on the structure of the main DataFrame.

        This method takes the positions and velocities of hydrogen atoms
        (HN3) and prepares a DataFrame for each APTES molecule based on
        the provided positions and velocities.

        Parameters:
            h_positions (Dict[str, Dict[int, np.ndarray]]): A dictionary
                containing hydrogen positions for each APTES molecule.
                The dictionary has APTES names as keys, and each value is
                another dictionary with integer keys (atom IDs) and numpy
                arrays representing the 3D coordinates (x, y, z) of the
                hydrogen atom.
            h_velocities (Dict[str, Dict[int, np.ndarray]]): A dictionary
                containing hydrogen velocities for each APTES molecule.
                The dictionary has APTES names as keys, and each value is
                another dictionary with integer keys (atom IDs) and numpy
                arrays representing the 3D velocities (vx, vy, vz) of the
                hydrogen atom.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing DataFrames
            for each APTES molecule, with columns ['residue_number',
            'residue_name', 'atom_name', 'atom_id', 'x', 'y', 'z', 'vx',
            'vy', 'vz'], where each row represents a hydrogen atom with
            its respective properties.

        Example:
            h_positions = {'APTES1': {1: np.array([1.0, 2.0, 3.0]),
                                      2: np.array([4.0, 5.0, 6.0])},
                           'APTES2': {1: np.array([7.0, 8.0, 9.0]),
                                      2: np.array([10.0, 11.0, 12.0])}}

            h_velocities = {'APTES1': {1: np.array([0.1, 0.2, 0.3]),
                                       2: np.array([0.4, 0.5, 0.6])},
                            'APTES2': {1: np.array([0.7, 0.8, 0.9]),
                                       2: np.array([0.10, 0.11, 0.12])}}

            result = prepare_hydrogens(h_positions, h_velocities)

            Output:
            {'APTES1': residue_number residue_name atom_name atom_id\
                    x    y    z   vx   vy   vz
                       0  1  APTES  HN3  1  1.0  2.0  3.0  0.1  0.2  0.3
                       1  2  APTES  HN3  2  4.0  5.0  6.0  0.4  0.5  0.6,
             'APTES2': residue_number residue_name atom_name atom_id\
                    x    y    z   vx   vy   vz
                         0  1  APTES  HN3  1  7.0  8.0  9.0  0.7  0.8  0.9
                         1  2  APTES  HN3  2 10.0 11.0 12.0 0.10 0.11 0.12
                         }}
        """
        cols: list[str] = \
            ['residue_number', 'residue_name', 'atom_name', 'atom_id',
             'x', 'y', 'z', 'vx', 'vy', 'vz']
        hn3_atoms_dict: dict[str, np.ndarray] = {}
        for aptes, positions in h_positions.items():
            hn3_atoms_i: pd.DataFrame = pd.DataFrame(columns=cols)
            for i, (key, pos) in enumerate(positions.items()):
                atom_id = i+1
                velo = h_velocities[aptes][int(key)]
                hn3_atoms_i.loc[i] = \
                    [key, aptes, 'HN3', atom_id, pos[0],
                     pos[1], pos[2], velo[0], velo[1], velo[2]]
            hn3_atoms_dict[aptes] = hn3_atoms_i
        return hn3_atoms_dict


class UpdateSolDf:
    """preparing water (SOL) residue for updating. Nothing needs to be
    change"""

    update_waters: pd.DataFrame  # Updated APTES df

    def __init__(self,
                 atoms: pd.DataFrame  # All SOL atoms
                 ) -> None:
        self.update_waters = self.update_water_df(atoms)

    @staticmethod
    def update_water_df(atoms: pd.DataFrame  # All SOL atoms
                        ) -> pd.DataFrame:
        """update water atoms if needed to be updated"""
        length = len(atoms.index) // 3
        llist = mk_atom_id_cycle(length, 1803)
        print(llist)
        print(len(llist))
        return atoms


class UpdateCorDf:
    """preparing COR residue for updating. Nothing needs to be
    change"""

    update_cor: dict[str, pd.DataFrame]  # Updated COR df

    def __init__(self,
                 atoms: dict[str, pd.DataFrame]  # All the cores atoms
                 ) -> None:
        self.update_cor = self.update_cor_df(atoms)

    @staticmethod
    def update_cor_df(atoms: dict[str, pd.DataFrame]  # All the cores atoms
                      ) -> dict[str, pd.DataFrame]:
        """update core atoms if needed to be updated"""
        id_updated_cores: dict[str, pd.DataFrame] = {}
        for cor, item in atoms.items():
            atom_ids: list[int] = [i+1 for i in range(len(item.index))]
            df_c: pd.DataFrame = item.copy()
            df_c['atom_id'] = atom_ids
            id_updated_cores[cor] = df_c
            df_c.to_csv(f'{cor}.test', sep=' ')
            del df_c
        return id_updated_cores


class UpdateOdaDf:
    """preparing ODA residue for updating.The residue index should be
    changed"""

    update_oda: pd.DataFrame  # Updated ODA df

    def __init__(self,
                 atoms: pd.DataFrame,  # All ODA atoms
                 protonation_nr: int  # Number of protonation
                 ) -> None:
        self.update_oda = self.update_oda_df(atoms, protonation_nr)

    @staticmethod
    def update_oda_df(atoms: pd.DataFrame,  # All ODA atoms
                      protonation_nr: int  # Number of protonation
                      ) -> pd.DataFrame:
        """update ODA atoms if needed to be updated"""
        df_c: pd.DataFrame = atoms.copy()
        updated_res: list[int] = atoms['residue_number'] + protonation_nr
        atoms_id: list[int] = [i+1 for i in range(len(atoms.index))]
        df_c['residue_number'] = updated_res
        df_c['atom_id'] = atoms_id
        return df_c


class UpdateOilDf:
    """preparing D10 residue for updating.The residue index should be
    changed"""

    update_oil: pd.DataFrame  # Updated D10 df

    def __init__(self,
                 atoms: pd.DataFrame,  # All oil atoms
                 protonation_nr: int  # Number of protonation
                 ) -> None:
        self.update_oil = self.update_oil_df(atoms, protonation_nr)

    @staticmethod
    def update_oil_df(atoms: pd.DataFrame,  # All oil atoms
                      protonation_nr: int  # Number of protonation
                      ) -> pd.DataFrame:
        """update oil atoms if needed to be updated"""
        df_c: pd.DataFrame = atoms.copy()
        updated_res: list[int] = atoms['residue_number'] + protonation_nr
        atoms_id: list[int] = [i+1 for i in range(len(atoms.index))]
        df_c['residue_number'] = updated_res
        df_c['atom_id'] = atoms_id
        return df_c


class UpdateIonDf:
    """preparing ION residue for updating. The new counterions should
    be added to the initial ones"""

    update_ion: pd.DataFrame  # Updated ION df

    def __init__(self,
                 atoms: pd.DataFrame,  # All ION atoms
                 ion_poses: list[np.ndarray],  # Position for the new ions
                 ion_velos: list[np.ndarray]  # Velocities for the new ions
                 ) -> None:
        self.update_ion = self.update_ion_df(atoms, ion_poses, ion_velos)

    def update_ion_df(self,
                      atoms: pd.DataFrame,  # All ION atoms
                      ion_poses: list[np.ndarray],  # Position for the new ions
                      ion_velos: list[np.ndarray]  # Velocities for the new ion
                      ) -> pd.DataFrame:
        """update ions atoms if needed to be updated"""
        final_res: int = atoms['residue_number'].iloc[-1]
        final_atom: int = atoms['atom_id'].iloc[-1]
        new_ions: pd.DataFrame = \
            self.prepare_ions(ion_poses, ion_velos, final_res, final_atom)
        updated_ions: pd.DataFrame = self.__append_ions(atoms, new_ions)
        return updated_ions

    @staticmethod
    def prepare_ions(ion_poses: list[np.ndarray],  # Ions positions
                     ion_velos: list[np.ndarray],  # Ions velocities
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


class UpdateResidues:
    """get all the dataframes as an object
    data: ionization.IonizationSol has following attributes:

        'atoms': pd.DataFrame -> All atoms from get_data.py

        'h_porotonations': dict[str, dict[int, np.ndarray]] -> All new
                           HN3 positions from protonation.py

        'h_velocities': dict[str, dict[int, np.ndarray]] -> All new
                        HN3 velocities from protonation.py

        'ion_poses': list[np.ndarray] -> Positoin for ions from
                     ionization.py

        'ion_velos': list[np.ndarray] -> Positoin for ions from
                     ionization.py

        'np_diameter': np.float64  -> Diameter of NP, based on APTES
                      positions from get_data.py

        'param': dict[str, typing.Any] -> Parameters for reading data
                 from read_param.py

        'pbc_box': str -> PBC of the system from get_data.py

        'residues_atoms': dict[str, pd.DataFrame] -> Atoms info for
                          each residue from get_data.py

        'title': str -> Name of the system from get_data.py

        'unprot_aptes_ind': dict[str, list[int]]  -> Index of APTES to
                            be protonated from get_data.py

        'unproton_aptes': dict[str, pd.DataFrame]  -> APTES which
                          should be protonated from get_data.py
    """

    # To append all the residues to the dict
    updated_residues: dict[str, pd.DataFrame] = {}
    # To append new atoms to the main atoms
    updated_atoms: pd.DataFrame
    # To return new atoms seperatly
    new_hn3: pd.DataFrame
    new_ions: pd.DataFrame
    # Final datafrme
    combine_residues: pd.DataFrame
    # Title and pbc of the system to write in the final file
    title: str
    pbc_box: str

    def __init__(self,
                 fname: str  # Name of the input file (pdb)
                 ) -> None:
        data = ionization.IonizationSol(fname)
        self.title = data.title
        self.pbc_box = data.pbc_box
        self.get_residues(data)
        combine_residues = self.concate_residues()
        self.combine_residues = self.__set_atom_id(combine_residues)

    @staticmethod
    def __set_atom_id(combine_residues: pd.DataFrame  # All the rsidues
                      ) -> pd.DataFrame:
        """set the atom_id for the all the atoms"""
        df_c: pd.DataFrame = combine_residues.copy()
        # Specify the limit for the atom IDs
        atom_id: list[int] = \
            mk_atom_id_cycle(len(combine_residues), start_id=1)
        # Calculate the number of cycles
        df_c['atom_id'] = atom_id
        df_c.to_csv('combine_residues', sep=' ')
        return df_c

    def concate_residues(self) -> pd.DataFrame:
        """concate all the residues in one dataframe, The order is
        very important. it should follow the order in the main file"""
        cols_order: list[str] = ['SOL', 'CLA', 'ODN', 'D10', 'COR', 'APT']
        # Concatenate DataFrames in the desired order
        combine_residues: pd.DataFrame = \
            pd.concat([self.updated_residues[col] for col in cols_order],
                      axis=0, ignore_index=True)
        return combine_residues

    def get_residues(self,
                     data: ionization.IonizationSol  # All the data
                     ) -> None:
        """get all the residues"""
        updated_aptes: dict[str, pd.DataFrame]  # All the updated aptes groups
        all_cores: dict[str, pd.DataFrame]  # All the cores atoms
        self.updated_residues['SOL'] = self.get_sol(data)
        self.updated_residues['D10'] = self.get_oil(data)
        self.updated_residues['CLA'] = self.get_ions(data)
        self.updated_residues['ODN'] = self.get_oda(data)
        updated_aptes, self.new_hn3 = self.get_aptes(data)
        for aptes, item in updated_aptes.items():
            self.updated_residues[aptes] = item
        all_cores = self.get_cor(data)
        for cores, item in all_cores.items():
            self.updated_residues[cores] = item

    @staticmethod
    def get_atoms(atoms: pd.DataFrame,  # Initial system
                  new_hn3: pd.DataFrame,  # New NH3 atoms
                  new_ions: pd.DataFrame,  # New Ions atoms
                  ) -> pd.DataFrame:
        """append the new atoms to the main dataframe with all atoms"""
        return pd.concat([atoms, new_hn3, new_ions])

    def get_aptes(self,
                  data: ionization.IonizationSol  # All the data
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """get updated aptes dataframe"""
        aptes_df_dict: dict[str, pd.DataFrame] = {}
        for aptes in data.param['aptes']:
            aptes_df_dict[aptes] = data.residues_atoms[aptes]
        updated_aptes = UpdateAptesDf(aptes_df_dict,
                                      data.h_porotonations,
                                      data.h_velocities)
        return updated_aptes.update_aptes, updated_aptes.new_nh3

    @staticmethod
    def get_sol(data: ionization.IonizationSol  # All the data
                ) -> pd.DataFrame:
        """return water residues"""
        updated_sol = UpdateSolDf(data.residues_atoms['SOL'])
        return updated_sol.update_waters

    @staticmethod
    def get_oil(data: ionization.IonizationSol  # All the data
                ) -> pd.DataFrame:
        """return oil residues"""
        updated_oils = UpdateOilDf(data.residues_atoms['D10'],
                                   int(len(data.ion_poses)))
        return updated_oils.update_oil

    @staticmethod
    def get_cor(data: ionization.IonizationSol  # All the data
                ) -> pd.DataFrame:
        """return core atoms of NP residues"""
        cores_atoms: dict[str, pd.DataFrame] = {}  # All the cores atoms
        for cor in data.param['cores']:
            cores_atoms[cor] = data.residues_atoms[cor]
        updated_cors = UpdateCorDf(cores_atoms)
        return updated_cors.update_cor

    @staticmethod
    def get_ions(data: ionization.IonizationSol  # All the data
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """get updated ions data frame"""
        updated_ions = UpdateIonDf(data.residues_atoms['CLA'],
                                   data.ion_poses,
                                   data.ion_velos)
        return updated_ions.update_ion

    @staticmethod
    def get_oda(data: ionization.IonizationSol  # All the data
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """get updated ions data frame"""
        updated_oda = UpdateOdaDf(data.residues_atoms['ODN'],
                                  int(len(data.ion_poses)))
        return updated_oda.update_oda


# Helper function to update index in gro fasion
def mk_atom_id_cycle(list_len: int,  # Size of the list,
                     start_id: int,  # First index of the residues
                     id_limit=99999  # Limit of the cycle
                     ) -> list[int]:
    """
    Generate a list of unique atom IDs in a custom cycle.

    This function generates a list of unique integers that follows a
    custom cycle pattern for atom IDs. The first cycle starts from the
    provided 'start_id' and goes up to 'id_limit', and subsequent
    cycles continue from 0 to 'id_limit'.

    Parameters:
        list_len (int): The desired size of the list to be generated.
                        The function will generate unique atom IDs
                        until reaching this size.
        start_id (int, optional): The starting value for the first
                                  cycle. The default value is 1.
        id_limit (int, optional): The upper limit of the atom ID cycle.
                                  The default value is 99999.

    Returns:
        List[int]: A list of unique atom IDs in the custom cycle.

    Example:
        >>> mk_atom_id_cycle(5)
        [1, 2, 3, 4, 5]
        >>> mk_atom_id_cycle(8, start_id=10)
        [10, 11, 12, 13, 14, 0, 1, 2]
        >>> mk_atom_id_cycle(12, start_id=100, id_limit=105)
        [100, 101, 102, 103, 104, 105, 0, 1, 2, 3, 4, 5]
    """
    counter = 0
    atoms_id = []
    while counter < list_len:
        if counter == 0:
            cycle_i = start_id
            cycle_f = id_limit + 1 - start_id
        else:
            cycle_i = 0
            cycle_f = id_limit + 1
        slice_i = [item + cycle_i for item in range(cycle_f)]
        counter += len(slice_i)
        atoms_id.extend(slice_i)
        del slice_i
    return atoms_id[:list_len]


if __name__ == '__main__':
    UpdateResidues(sys.argv[1])
