"""To properly format GRO files, velocities columns must be included
in the input file. It is essential to index the atoms and residues
correctly. The indexing of the atoms and residues does not start at
zero; the index of the first residue is actually one greater than the
last residue written in the input file. Additionally, IONs will
increase the number of residues and atom numbers, but the new HN3 only
increases the number of atoms.
All the dataframes are needed. Everthing should be updated.
    Classes:
        UpdateBaseDf: This is a base class for updating dataframes.
                      It includes methods to update the indices of
                      atoms and residues in a given dataframe.
        UpdateNanoParticle: This class updates nanoparticles by
                            stacking core atoms and APTES atoms, fixing
                            their indices accordingly.
        UpdateAptesDf: This class updates the APTES dataframe by
                       adding new HN3 atoms based on provided positions
                       and velocities.
        UpdateCorDf: This class prepares the COR residue for updating.
                     It doesn't need any changes.
        UpdateSolDf: This class prepares the SOL (water) residue for
                     updating. It updates the atom and residue indices.
        UpdateOilDf: This class prepares the D10 residue for updating.
                     It updates the atom and residue indices.
        UpdateOdaDf: This class prepares the ODA residue for updating.
                     It updates the atom and residue indices.
        UpdateIonDf: This class prepares the ION residue for updating.
                     It updates the atom and residue indices and adds
                     new counterions to the initial ones.
        UpdateResidues: This is the main class that coordinates the
                        updating of all residues. It includes methods
                        to update water, oil, APTES, and other
                        nanoparticle residues.

    Helper Functions:
        mk_atom_id_cycle: This function generates a list of unique
                          atom IDs in a custom cycle, starting from a
                          specified value and repeating within a
                          specified limit.
        repeat_items: This function repeats each item in a list a given
                      number of times.

"""

import sys
import typing
import json
import numpy as np
import pandas as pd
import logger
import my_tools
import ionization
from colors_text import TextColor as bcolors
if typing.TYPE_CHECKING:
    from ionization import IonizationSol


class UpdateBaseDf:
    """Base class for updating dataframes."""

    def __init__(self,
                 atoms: pd.DataFrame,  # Atoms to update their indices
                 first_res: int,  # The First index to start with
                 first_atom: int,  # The First index to start with
                 atoms_per_res: int,  # The First index to start with
                 ) -> None:
        self.first_res = first_res
        self.first_atom = first_atom
        self.atoms_per_res = atoms_per_res
        self.update_df = self.mk_update_df(atoms)
        self.last_res = self.update_df['residue_number'].iloc[-1]
        self.last_atom = self.update_df['atom_id'].iloc[-1]
        self.nr_atoms: int = int(len(self.update_df))
        self.nr_residues: int = self.nr_atoms // self.atoms_per_res

    def mk_update_df(self,
                     atoms: pd.DataFrame  # Atoms to update their indices
                     ) -> pd.DataFrame:
        """Update the dataframe with the given atoms.

        Parameters:
            atoms (pd.DataFrame): All atoms dataframe.

        Returns:
            pd.DataFrame: Updated dataframe.
        """
        atoms_c = atoms.copy()
        nr_atoms = len(atoms.index)
        updated_res_id = self.update_residue_index(nr_atoms)
        atoms_c.loc[:, 'residue_number'] = updated_res_id
        updated_atom_id = self.update_atom_index(nr_atoms)
        atoms_c.loc[:, 'atom_id'] = updated_atom_id
        return atoms_c

    def update_atom_index(self,
                          nr_atoms: int  # Number of the atoms in the df
                          ) -> list[int]:
        """Update atom indices.

        Parameters:
            nr_atoms (int): Number of all atoms.

        Returns:
            list[int]: List of updated atom_id values.
        """
        return mk_atom_id_cycle(nr_atoms, self.first_atom)

    def update_residue_index(self,
                             nr_atoms: int  # Number of the atoms in the df
                             ) -> list[int]:
        """Prepare the residues index.

        Parameters:
            nr_atoms (int): Number of all atoms.

        Returns:
            list[int]: List of updated residue_number values.
        """
        list_index = \
            mk_atom_id_cycle(nr_atoms // self.atoms_per_res, self.first_res)
        return repeat_items(list_index, self.atoms_per_res)


class UpdateNanoParticle:
    """
    Update nanoparticles by staking cores aptes by fixing the indicies
    """
    def __init__(self,
                 aptes_df: pd.DataFrame,  # Aptes' atoms
                 cores_df: pd.DataFrame,  # Cores' atoms
                 ) -> None:
        self.nanop_updated: pd.DataFrame = \
            self.stack_nano_p(aptes_df, cores_df)

    def stack_nano_p(self,
                     aptes_df: pd.DataFrame,  # Aptes' atoms
                     cores_df: pd.DataFrame  # Cores' atoms
                     ) -> pd.DataFrame:
        """making the total nanoparticle"""
        cor_lst_atom: int = cores_df.iloc[-1]['atom_id']
        aptes_df = self.update_aptes_atom_id(aptes_df, cor_lst_atom)
        return pd.concat([cores_df, aptes_df])

    @staticmethod
    def update_aptes_atom_id(aptes_df: pd.DataFrame,  # Aptes atoms
                             cor_lst_atom: int
                             ) -> pd.DataFrame:
        """update the indices of the aptes"""
        atoms_c: pd.DataFrame = aptes_df.copy()
        atoms_c['atom_id'] += cor_lst_atom
        return atoms_c


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
                pd.concat([item, hn3_atoms[aptes]], ignore_index=True)
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


class UpdateCorDf:
    """preparing COR residue for updating. Nothing needs to be
    change"""

    update_cor: dict[str, pd.DataFrame]  # Updated COR df

    def __init__(self,
                 atoms: dict[str, pd.DataFrame],  # All the cores atoms,
                 debug: bool
                 ) -> None:
        self.update_cor = self.update_cor_df(atoms, debug)

    @staticmethod
    def update_cor_df(atoms: dict[str, pd.DataFrame],  # All the cores atoms
                      debug: bool
                      ) -> dict[str, pd.DataFrame]:
        """update core atoms if needed to be updated"""
        id_updated_cores: dict[str, pd.DataFrame] = {}
        for cor, item in atoms.items():
            atom_ids: list[int] = [i+1 for i in range(len(item.index))]
            df_c: pd.DataFrame = item.copy()
            df_c['atom_id'] = atom_ids
            id_updated_cores[cor] = df_c
            if debug != 'None':
                df_c.to_csv(f'{cor}_update_atom_id.debug', sep=' ')
            del df_c
        return id_updated_cores


class UpdateSolDf(UpdateBaseDf):
    """Class for preparing water (SOL) residue for updating."""

    def __init__(self,
                 atoms: pd.DataFrame,
                 debug: bool
                 ) -> None:
        name = 'sol'
        first_res = atoms['residue_number'].iloc[0]
        first_atom = 1
        atoms_per_res = 3
        super().__init__(atoms, first_res, first_atom, atoms_per_res)
        if debug != 'None':
            self.update_df.to_csv(f'{name}_res_update.debug', sep=' ')


class UpdateOilDf(UpdateBaseDf):
    """Class for preparing D10 residue for updating."""

    def __init__(self,
                 atoms: pd.DataFrame,
                 sol_last_res: int,
                 sol_last_atom: int,
                 debug: bool
                 ) -> None:
        name = 'oil'
        first_res = sol_last_res + 1
        first_atom = sol_last_atom + 1
        atoms_per_res = 32
        super().__init__(atoms, first_res, first_atom, atoms_per_res)
        if debug != 'None':
            self.update_df.to_csv(f'{name}_res_update.debug', sep=' ')


class UpdateOdaDf(UpdateBaseDf):
    """preparing ODA residue for updating.The residue index should be
    changed"""

    def __init__(self,
                 atoms: pd.DataFrame,
                 last_res: int,
                 last_atom: int,
                 debug: bool
                 ) -> None:
        name = 'oda'
        first_res = last_res + 1
        first_atom = last_atom + 1
        atoms_per_res = 59
        super().__init__(atoms, first_res, first_atom, atoms_per_res)
        if debug != 'None':
            self.update_df.to_csv(f'{name}_res_update.debug', sep=' ')


class UpdateOdmDf(UpdateBaseDf):
    """preparing ODA residue for updating.The residue index should be
    changed"""

    def __init__(self,
                 atoms: pd.DataFrame,
                 last_res: int,
                 last_atom: int,
                 debug: bool
                 ) -> None:
        name = 'odm'
        first_res = last_res + 1
        first_atom = last_atom + 1
        atoms_per_res = 58
        super().__init__(atoms, first_res, first_atom, atoms_per_res)
        if debug != 'None':
            self.update_df.to_csv(f'{name}_res_update.debug', sep=' ')


class UpdatePotDf(UpdateBaseDf):
    """preparing ODA residue for updating.The residue index should be
    changed"""

    def __init__(self,
                 atoms: pd.DataFrame,
                 ion_last_res: int,
                 ion_last_atom: int,
                 debug: bool
                 ) -> None:
        name = 'pot'
        first_res = ion_last_res + 1
        first_atom = ion_last_atom + 1
        atoms_per_res = 1
        super().__init__(atoms, first_res, first_atom, atoms_per_res)
        if debug != 'None':
            self.update_df.to_csv(f'{name}_res_update.debug', sep=' ')


class UpdateIonDf(UpdateBaseDf):
    """preparing ION residue for updating. The new counterions should
    be added to the initial ones"""

    def __init__(self,
                 atoms: pd.DataFrame,  # All ION atoms
                 ion_poses: list[np.ndarray],  # Position for the new ions
                 ion_velos: list[np.ndarray],  # Velocities for the new ions
                 last_res: int,
                 last_atom: int,
                 debug: bool
                 ) -> None:
        all_ion = self.update_ion_df(atoms, ion_poses, ion_velos)
        name = 'ion'
        first_res = last_res + 1
        first_atom = last_atom + 1
        atoms_per_res = 1
        super().__init__(all_ion, first_res, first_atom, atoms_per_res)
        if debug != 'None':
            self.update_df.to_csv(f'{name}_res_update.debug', sep=' ')

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
        all_ions: pd.DataFrame = self.__append_ions(atoms, new_ions)
        return all_ions

    @staticmethod
    def __append_ions(ions: pd.DataFrame,  # initial ions
                      new_ions: pd.DataFrame  # New ions atoms to append
                      ) -> pd.DataFrame:
        """append new ions atoms to the main df"""
        return pd.concat([ions, new_ions], ignore_index=False)

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


class UpdateResidues:
    """get all the dataframes as an object
    data: 'IonizationSol' has following attributes:

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

    # Message to pass for logging and writing
    info_msg: str = 'Messages from UpdateResidues:\n'
    # To append all the residues to the dict
    updated_residues: dict[str, pd.DataFrame] = {}
    # To append new atoms to the main atoms
    updated_atoms: pd.DataFrame
    # To return new atoms seperatly
    new_hn3: pd.DataFrame
    new_ions: pd.DataFrame
    # Final datafrme
    combine_residues: pd.DataFrame
    nr_atoms_residues: dict[str, dict[str, int]] = {}
    # Title and pbc of the system to write in the final file
    title: str
    pbc_box: str

    def __init__(self,
                 fname: str,  # Name of the input file (pdb)
                 log
                 ) -> None:
        data = ionization.IonizationSol(fname, log)
        self.title = data.title
        self.pbc_box = data.pbc_box
        self.get_residues(data)
        combine_residues = self.concate_residues(data.param)
        self.combine_residues = self.__set_atom_id(combine_residues,
                                                   data.param['DEBUG'])
        self.write_log_msg(log)
        # last mins add!!
        self.param = data.param

    @staticmethod
    def __set_atom_id(combine_residues: pd.DataFrame,  # All the rsidues
                      debug: bool
                      ) -> pd.DataFrame:
        """set the atom_id for the all the atoms"""
        df_c: pd.DataFrame = combine_residues.copy()
        # Specify the limit for the atom IDs
        atom_id: list[int] = \
            mk_atom_id_cycle(len(combine_residues), start_id=1)
        # Calculate the number of cycles
        df_c['atom_id'] = atom_id
        if debug != 'None':
            df_c.to_csv('combine_residues.debug', sep=' ')
        return df_c

    def concate_residues(self,
                         param: dict[str, typing.Any]
                         ) -> pd.DataFrame:
        """concate all the residues in one dataframe, The order is
        very important. it should follow the order in the main file"""
        cols_order: list[str] = ['SOL', 'D10', 'ODN', 'CLA', 'POT', 'ODM']
        for item in param['itp_files']:
            np_name: str = my_tools.drop_string(item, '.itp')
            cols_order.append(np_name)
        # Drop unesist residues
        for res in ['ODN', 'POT', 'ODM']:
            if res not in self.updated_residues:
                cols_order.remove(res)
        # Concatenate DataFrames in the desired order
        combine_residues: pd.DataFrame = \
            pd.concat([self.updated_residues[col] for col in cols_order],
                      axis=0, ignore_index=True)

        return combine_residues

    def get_residues(self,
                     data: 'IonizationSol'  # All the data
                     ) -> None:
        """get all the residues"""

        update_sol: UpdateSolDf = self.get_sol(data)
        self.updated_residues['SOL'] = update_sol.update_df
        self.nr_atoms_residues['SOL'] = {'nr_atoms': update_sol.nr_atoms,
                                         'nr_residues': update_sol.nr_residues}

        update_oil: UpdateOilDf = self.get_oil(data,
                                               update_sol.last_res,
                                               update_sol.last_atom)
        self.updated_residues['D10'] = update_oil.update_df
        self.nr_atoms_residues['D10'] = {'nr_atoms': update_oil.nr_atoms,
                                         'nr_residues': update_oil.nr_residues}
        last_res: int = update_oil.last_res
        last_atom: int = update_oil.last_atom

        try:
            update_oda: UpdateOdaDf = self.get_oda(data, last_res, last_atom)
            self.updated_residues['ODN'] = update_oda.update_df
            self.nr_atoms_residues['ODN'] = \
                {'nr_atoms': update_oda.nr_atoms,
                 'nr_residues': update_oda.nr_residues}
            last_res = update_oda.last_res
            last_atom = update_oda.last_atom
        except KeyError:
            pass

        update_ion: UpdateIonDf = self.get_ions(data, last_res, last_atom)
        self.updated_residues['CLA'] = update_ion.update_df
        self.nr_atoms_residues['CLA'] = {'nr_atoms': update_ion.nr_atoms,
                                         'nr_residues': update_ion.nr_residues}

        last_res = update_ion.last_res
        last_atom = update_ion.last_atom

        try:
            update_pot: UpdatePotDf = self.get_pots(data,
                                                    update_ion.last_res,
                                                    update_ion.last_atom)
            self.updated_residues['POT'] = update_pot.update_df
            self.nr_atoms_residues['POT'] = \
                {'nr_atoms': update_pot.nr_atoms,
                 'nr_residues': update_pot.nr_residues}
            last_res = update_pot.last_res
            last_atom = update_pot.last_atom
        except KeyError:
            pass

        try:
            update_odm: UpdateOdmDf = self.get_odm(data, last_res, last_atom)
            self.updated_residues['ODM'] = update_odm.update_df
            self.nr_atoms_residues['ODM'] = \
                {'nr_atoms': update_odm.nr_atoms,
                 'nr_residues': update_odm.nr_residues}
        except KeyError:
            pass

        updated_aptes: dict[str, pd.DataFrame]  # All the updated aptes groups
        updated_aptes, self.new_hn3 = self.get_aptes(data)
        self.add_nanoparticles_to_updated_residues(data,
                                                   update_ion,
                                                   updated_aptes)
        self.get_nr_atoms_residues_in_np(data.param)

        self.info_msg += '\tNumber of residues and atoms in each residue:\n'
        self.info_msg += json.dumps(self.nr_atoms_residues, indent=8)
        self.info_msg += '\n'

    def add_nanoparticles_to_updated_residues(self,
                                              data: 'IonizationSol',
                                              update_ion: UpdateIonDf,
                                              updated_aptes: dict[str,
                                                                  pd.DataFrame]
                                              ) -> None:
        """
        Update nanoparticles by updating residues and atoms indices
        """
        updated_np_dict: dict[str, pd.DataFrame] = \
            self.update_nanoparticles(data, updated_aptes)
        last_residue: int = update_ion.last_res
        last_atom: int = update_ion.last_atom + 1
        for nano_p, item in updated_np_dict.items():
            df_c: pd.DataFrame = item.copy()
            nr_atoms: int = len(item.index)
            updated_atom_id: list[int] = \
                mk_atom_id_cycle(list_len=nr_atoms, start_id=last_atom)
            df_c.loc[:, 'atom_id'] = updated_atom_id
            updated_res_id: list[int] = df_c['residue_number'] + last_residue
            if not any(num > 99999 for num in updated_res_id):
                df_c.loc[:, 'residue_number'] += last_residue
                self.updated_residues[nano_p] = df_c
                if data.param['DEBUG'] != 'None':
                    df_c.to_csv(f'{nano_p}_res_uodate.debug', sep=' ')
            else:
                sys.exit(f'\n{bcolors.FAIL}{self.__module__}:\n'
                         'Error: Some members in the list are bigger '
                         'than the 99999 value. Must update the codes!'
                         f'{bcolors.ENDC}\n')
            last_atom = df_c.iloc[-1]['atom_id'] + 1
            last_residue = int(np.max(df_c['residue_number']))

    def update_nanoparticles(self,
                             data: 'IonizationSol',
                             updated_aptes:  dict[str, pd.DataFrame]
                             ) -> dict[str, pd.DataFrame]:
        """update nanoparticles"""
        all_cores: dict[str, pd.DataFrame]  # All the cores atoms
        all_cores = self.get_cor(data)
        updated_np_dict: dict[str, pd.DataFrame] = {}
        for i in range(len(data.param['itp_files'])):
            aptes: str = data.param['aptes'][i]
            cores: str = data.param['cores'][i]
            np_atoms = UpdateNanoParticle(
                aptes_df=updated_aptes[aptes], cores_df=all_cores[cores])
            np_name: str = \
                my_tools.drop_string(data.param['itp_files'][i], '.itp')
            updated_np_dict[np_name] = np_atoms.nanop_updated
        return updated_np_dict

    @staticmethod
    def get_atoms(atoms: pd.DataFrame,  # Initial system
                  new_hn3: pd.DataFrame,  # New NH3 atoms
                  new_ions: pd.DataFrame,  # New Ions atoms
                  ) -> pd.DataFrame:
        """append the new atoms to the main dataframe with all atoms"""
        return pd.concat([atoms, new_hn3, new_ions])

    def get_aptes(self,
                  data: 'IonizationSol'  # All the data
                  ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """get updated aptes dataframe"""
        aptes_df_dict: dict[str, pd.DataFrame] = {}
        for aptes in data.param['aptes']:
            aptes_df_dict[aptes] = data.residues_atoms[aptes]
        updated_aptes = UpdateAptesDf(aptes_df_dict,
                                      data.h_porotonations,
                                      data.h_velocities,
                                      )
        return updated_aptes.update_aptes, updated_aptes.new_nh3

    @staticmethod
    def get_cor(data: 'IonizationSol'  # All the data
                ) -> pd.DataFrame:
        """return core atoms of NP residues"""
        cores_atoms: dict[str, pd.DataFrame] = {}  # All the cores atoms
        for cor in data.param['cores']:
            cores_atoms[cor] = data.residues_atoms[cor]
        updated_cors = UpdateCorDf(cores_atoms, data.param['DEBUG'])
        return updated_cors.update_cor

    @staticmethod
    def get_sol(data: 'IonizationSol'  # All the data
                ) -> pd.DataFrame:
        """return water residues"""
        updated_sol = UpdateSolDf(data.residues_atoms['SOL'],
                                  data.param['DEBUG'])
        return updated_sol

    @staticmethod
    def get_oil(data: 'IonizationSol',  # All the data
                sol_last_res: int,  # Last residue index in water
                sol_last_atom: int  # Last atom index in water
                ) -> pd.DataFrame:
        """return oil residues"""
        updated_oils = UpdateOilDf(data.residues_atoms['D10'],
                                   sol_last_res,
                                   sol_last_atom,
                                   data.param['DEBUG'])
        return updated_oils

    @staticmethod
    def get_oda(data: 'IonizationSol',  # All the data
                oil_last_res: int,  # Last residue index in water
                oil_last_atom: int  # Last atom index in water
                ) -> UpdateOdaDf:
        """get updated ions data frame"""
        updated_oda = UpdateOdaDf(data.residues_atoms['ODN'],
                                  oil_last_res,
                                  oil_last_atom,
                                  data.param['DEBUG']
                                  )
        return updated_oda

    @staticmethod
    def get_odm(data: 'IonizationSol',  # All the data
                last_res: int,  # Last residue index in water
                last_atom: int  # Last atom index in water
                ) -> UpdateOdmDf:
        """get updated ions data frame"""
        updated_odm = UpdateOdmDf(data.residues_atoms['ODN'],
                                  last_res,
                                  last_atom,
                                  data.param['DEBUG']
                                  )
        return updated_odm

    @staticmethod
    def get_ions(data: 'IonizationSol',  # All the data
                 oda_last_res: int,
                 oda_last_atom: int
                 ) -> UpdateIonDf:
        """get updated ions data frame"""
        updated_ions = UpdateIonDf(data.residues_atoms['CLA'],
                                   data.ion_poses,
                                   data.ion_velos,
                                   oda_last_res,
                                   oda_last_atom,
                                   data.param['DEBUG']
                                   )
        return updated_ions

    @staticmethod
    def get_pots(data: 'IonizationSol',  # All the data
                 ion_last_res: int,
                 ion_last_atom: int
                 ) -> UpdatePotDf:
        """get updated POT ion data frame"""
        updated_ions = UpdatePotDf(data.residues_atoms['POT'],
                                   ion_last_res,
                                   ion_last_atom,
                                   data.param['DEBUG']
                                   )
        return updated_ions

    def get_nr_atoms_residues_in_np(self,
                                    param: dict[str, typing.Any]
                                    ) -> None:
        """
        Count unique numbers in 'atom_id' and 'residue_number' columns
        for each nanoparticles.
        """

        for nano_p in param["itp_files"]:
            np_name: str = my_tools.drop_string(nano_p, '.itp')
            df_i: pd.DataFrame = self.updated_residues[np_name]
            # Count unique numbers in 'atom_id' and 'residues_number' columns
            atom_id_count = len(df_i['atom_id'])
            residues_number_count = df_i['residue_number'].nunique()

            # Store the counts in the result dictionary
            self.nr_atoms_residues[nano_p] = {
                'nr_atoms': atom_id_count,
                'nr_residues': residues_number_count
            }

    def write_log_msg(self,
                      log: logger.logging.Logger  # Name of the output file
                      ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)
        print(f'{bcolors.OKBLUE}{UpdateResidues.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')


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


def repeat_items(lst: list[int],  # List index which should repeat for residues
                 n_atom_per_res: int  # Number of the atoms in residues
                 ) -> list:
    """
    Repeat each item in the list 'n' times.

    This function takes a list and repeats each item in the list 'n'
    times, creating a new list with the repeated items.

    Parameters:
        lst (list): The input list to repeat items from.
        n (int): The number of times each item should be repeated.

    Returns:
        List: A new list with each item repeated 'n' times.

    Example:
        >>> repeat_items([1, 2, 3], 2)
        [1, 1, 2, 2, 3, 3]
        >>> repeat_items(['a', 'b', 'c'], 3)
        ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c']
    """
    repeated_list = [item for item in lst for _ in range(n_atom_per_res)]
    return repeated_list


if __name__ == '__main__':
    UpdateResidues(sys.argv[1], log=logger.setup_logger('update.log'))
