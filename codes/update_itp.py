"""
The script updates the nanoparticles "itp" file using the HN3 atoms set
by update_residue_gro.py. To do this, we need to add the correct
information for the name and charge for the new HN3 in the atoms
section.
Additionally, for chains that have been protonated, It also updates
the name and charges for N, HN1, and HN2 to match the new values.
The script needs to add bonds between N and the new HN3. Since force-
field values are set in different files, we only need to update the
sections for the extra H in each chain, including angles and dihedrals.

There are several classes defined in the script, including:
    UpdateItp, UpdateDihedral, UpdateAngle, UpdateBond, UpdateAtom,
    WrapperUpdateItp, and StandAlone.

The UpdateItp class is the main class responsible for updating the ITP
    file. It initializes with the name of the ITP file, information
    about new HN3 atoms, and the names of APTES and CORE residues.
The UpdateDihedral, UpdateAngle, UpdateBond, and UpdateAtom classes
    are helper classes used by UpdateItp to update specific sections
    (dihedrals, angles, bonds, atoms) in the ITP file.
The WrapperUpdateItp class acts as a wrapper to update multiple ITP
    files, each corresponding to a different set of HN3 atoms.
The StandAlone class is used to generate random positions and velocities
    for the new HN3 atoms for testing purposes.
The script reads an ITP file, updates the specified sections, and writes
    the updated ITP to a new file with the name 'APT_COR_updated.itp'.

To use this script, you would need to provide it with the input ITP
file name and the information about the new HN3 atoms. The StandAlone
class is used to generate test data for new HN3 atoms when running the
script independently.
"""

import sys
import typing
import numpy as np
import pandas as pd
import logger
import my_tools
import itp_to_df as itp
from colors_text import TextColor as bcolors


class UpdateItp(itp.Itp):
    """
    Updating the itp file for each section.
        The object from itp class
        - atoms: pd.DataFrame -> Section of initial itp file
        - bonds: pd.DataFrame -> Section of initial itp file
        - angles: pd.DataFrame -> Section of initial itp file
        - dihedrals: pd.DataFrame -> Section of initial itp file
        - molecules: pd.DataFrame -> Section of initial itp file
    """

    atoms_updated: pd.DataFrame  # Updated atoms section
    bonds_updated: pd.DataFrame  # Updated bonds section
    angles_updated: pd.DataFrame  # Updated angles section
    dihedrals_updated: pd.DataFrame  # Updated dihedrals section

    def __init__(self,
                 fname: str,  # Name of the itp file
                 hn3: pd.DataFrame,  # Information for new NH3 atoms
                 aptes: str,  # Name of the APTES branches
                 core: str  # Name of the CORE residues
                 ) -> None:
        super().__init__(fname)
        self.update_itp(hn3, aptes, core)

    def update_itp(self,
                   hn3: pd.DataFrame,  # NH3 atoms
                   aptes: str,  # Name of the APTES branches
                   core: str  # Name of the CORE residues
                   ) -> None:
        """get all the data and return final df"""
        up_atoms = UpdateAtom(self.atoms, hn3, aptes, core)
        up_bonds = UpdateBond(self.bonds, hn3, up_atoms.atoms_updated)
        up_angles = UpdateAngle(self.angles, hn3, up_atoms.atoms_updated)
        up_dihedrals = \
            UpdateDihedral(self.dihedrals, hn3, up_atoms.atoms_updated)
        self.atoms_updated = up_atoms.atoms_updated
        self.bonds_updated = up_bonds.bonds_updated
        self.angles_updated = up_angles.angles_updated
        self.dihedrals_updated = up_dihedrals.dihedrals_updated


class UpdateDihedral:
    """update dihedrals and return the updated section contian new HN3
    """

    dihedrals_updated: pd.DataFrame  # Updated section with new HN3

    def __init__(self,
                 dihedral_np: pd.DataFrame,  # Dihedral from itp file for NP
                 hn3: pd.DataFrame,  # New HN3 to add to the atoms
                 atoms: pd.DataFrame  # Updated APTES chains by UpAtom class
                 ) -> None:
        self.dihedrals_updated = self.update_dihedrals(dihedral_np, hn3, atoms)

    def update_dihedrals(self,
                         dihedral_np: pd.DataFrame,  # Dihedral from itp file
                         hn3: pd.DataFrame,  # New HN3 to add to the atoms
                         atoms: pd.DataFrame  # Updated APTES chains by UpAtom
                         ) -> pd.DataFrame:
        """update dihedral section"""
        # Find the dihedrals which contain N and HN3
        unique_dihedrals: pd.DataFrame = self.__get_dihedrals(dihedral_np)
        # Get the index of the new residues with new HN3 atoms
        new_proton_res: list[int] = UpdateAngle.get_hn3_index(hn3)
        new_dihedrals: pd.DataFrame = \
            self.mk_dihedrlas(atoms, new_proton_res, unique_dihedrals)
        return pd.concat([dihedral_np, new_dihedrals],
                         axis=0, ignore_index=True)

    @staticmethod
    def mk_dihedrlas(atoms: pd.DataFrame,  # Updated atoms with new HN3
                     new_proton_res: list[int],  # Index of the protonated APT
                     unique_dihedrals: pd.DataFrame  # Types of dihedral to mk
                     ) -> pd.DataFrame:
        """make dataframe from new dihedrals"""
        fourth_atoms: dict[str, int]  # Fourth atoms in dihedrals
        fourth_atoms = \
            UpdateAngle.get_atom_in_angdihd(unique_dihedrals,
                                            ignore_list=['N', 'HN3', 'CT'])
        res_dihedrals: list[pd.DataFrame] = []  # All the dihedrals
        for res in new_proton_res:
            df_res = atoms[atoms['resnr'] == res]
            res_dihedrals.append(
                UpdateDihedral.get_dihedral_res(df_res, fourth_atoms))
        return pd.concat(res_dihedrals)

    @staticmethod
    def get_dihedral_res(df_res: pd.DataFrame,  # Only one residues
                         fourth_atoms: dict[str, int]  # 4th atom and type of d
                         ) -> pd.DataFrame:
        """creat dihedrals for each residues which got new HN3"""
        n_index: int = \
            df_res.loc[df_res['atomname'] == 'N', 'atomnr'].values[0]
        hn3_index: int = \
            df_res.loc[df_res['atomname'] == 'HN3', 'atomnr'].values[0]
        ct_index: int = \
            df_res.loc[df_res['atomname'] == 'CT', 'atomnr'].values[0]
        columns: list[str] = ['typ', 'ai', 'aj', 'ak', 'ah', 'cmt', 'name']
        dihedral_res = pd.DataFrame(columns=columns)
        for i, (atom, funct) in enumerate(fourth_atoms.items()):
            fourth_id = \
                df_res.loc[df_res['atomname'] == atom, 'atomnr'].values[0]
            dihedral_name = f'{atom}-CT-N-HN3'
            dihedral_res.loc[i] = [funct, fourth_id, ct_index, n_index,
                                   hn3_index, ';', dihedral_name]
        return dihedral_res

    @staticmethod
    def __get_dihedrals(dihedral_np: pd.DataFrame  # All from itp file
                        ) -> pd.DataFrame:
        """get unique dataframe of all of those which involve N and HN3"""

        condition: pd.Series = dihedral_np['name'].str.contains('N-') & \
            dihedral_np['name'].str.contains('HN3')
        unique_angles: pd.DataFrame = \
            dihedral_np.loc[condition, ['name', 'typ']].drop_duplicates()
        return unique_angles


class UpdateAngle:
    """update angles section by adding all the needed angle which
    involve HN3"""

    angles_updated: pd.DataFrame  # Updated angles section with new angles

    def __init__(self,
                 angle_np: pd.DataFrame,  # Angles from the itp file for NP
                 hn3: pd.DataFrame,  # New HN3 to add to the atoms
                 atoms: pd.DataFrame  # Updated APTES chains by UpAtom class
                 ) -> None:
        self.angles_updated = self.update_angles(angle_np, hn3, atoms)

    def update_angles(self,
                      angle_np: pd.DataFrame,  # Angles from the itp file
                      hn3: pd.DataFrame,  # New HN3 to add to the atoms
                      atoms: pd.DataFrame  # Updated APTES chains by UpAtom
                      ) -> pd.DataFrame:
        """
        Update the angles section of the itp file with new angles
        involving HN3 atoms.

        Args:
            angle_np (pd.DataFrame): Angles from the itp file.
            hn3 (pd.DataFrame): New HN3 to add to the atoms.
            atoms (pd.DataFrame): Updated APTES chains by UpAtom class.

        Returns:
            pd.DataFrame: Updated angles section with new angles.
        """
        # Find the angles which involved N and HN3
        unique_angles: pd.DataFrame = self.__get_angles(angle_np)
        # Get the index of the residues with new HN3 atoms
        new_proton_res: list[int] = self.get_hn3_index(hn3)
        # Make angles for the new HN3s
        new_angles: pd.DataFrame = \
            self.mk_angels(atoms, new_proton_res, unique_angles)
        return pd.concat([angle_np, new_angles], axis=0, ignore_index=True)

    @staticmethod
    def mk_angels(atoms: pd.DataFrame,  # Updated APTES chains' atoms
                  new_proton_res: list[int],  # Index of the protonated APTES
                  unique_angles: pd.DataFrame  # Type of angle to create
                  ) -> pd.DataFrame:
        """
        Make dataframe for new angles involving HN3 atoms.

        Args:
            atoms (pd.DataFrame): Updated APTES chains' atoms.
            new_proton_res (list[int]): Index of the protonated APTES
            residues.
            unique_angles (pd.DataFrame): Type of angles to create.

        Returns:
            pd.DataFrame: DataFrame of new angles involving HN3 atoms.
        """
        third_atom_angle: dict[str, int]  # Third atoms name in each angle
        third_atom_angle = \
            UpdateAngle.get_atom_in_angdihd(unique_angles,
                                            ignore_list=['N', 'HN3'])
        res_angles: list[pd.DataFrame] = []  # Angels of each residue
        for res in new_proton_res:
            df_res = atoms[atoms['resnr'] == res]
            res_angles.append(
                UpdateAngle.__get_angles_res(df_res, third_atom_angle))
        return pd.concat(res_angles)

    @staticmethod
    def __get_angles_res(df_res: pd.DataFrame,  # Only one residues
                         third_atom_angle: dict[str, int]  # Atoms and type
                         ) -> pd.DataFrame:
        """
        Create angles for each residue involving HN3 atoms.

        Args:
            df_res (pd.DataFrame): DataFrame containing atoms of a
            single residue.
            third_atom_angle (dict[str, int]): Atoms and angle types.

        Returns:
            pd.DataFrame: DataFrame of angles involving HN3 atoms for
            the residue.
        """
        n_index: int = \
            df_res.loc[df_res['atomname'] == 'N', 'atomnr'].values[0]
        hn3_index: int = \
            df_res.loc[df_res['atomname'] == 'HN3', 'atomnr'].values[0]
        columns: list[str] = ['typ', 'ai', 'aj', 'ak', 'cmt', 'name']
        angle_res = pd.DataFrame(columns=columns)
        for i, (atom, funct) in enumerate(third_atom_angle.items()):
            third_id = \
                df_res.loc[df_res['atomname'] == atom, 'atomnr'].values[0]
            angle_name = f'{atom}-N-HN3'
            angle_res.loc[i] = \
                [funct, third_id, n_index, hn3_index, ';', angle_name]
        return angle_res

    @staticmethod
    def get_atom_in_angdihd(unique_angles: pd.DataFrame,  # Name & typ of angle
                            ignore_list: list[str]  # Atoms to ignore
                            ) -> dict[str, int]:
        """break down the names of the angles and return name of the
        atom which is not N or HN3, it can be used by dihedral class"""
        # Splitting the 'name' column and combining it with 'typ' column
        names: list[str]  # Name of the angles
        names = unique_angles['name'].to_list()
        types: list[int] = unique_angles['typ'].to_list()
        atom_names: list[list[str]] = [item.split('-') for item in names]
        third_atoms: list[str] = [item for l_item in atom_names for item
                                  in l_item if item not in ignore_list]
        return dict(zip(third_atoms, types))

    @staticmethod
    def get_hn3_index(hn3: pd.DataFrame  # New HN3 atoms
                      ) -> list[int]:
        """return list of all the residues"""
        return hn3['residue_number'].drop_duplicates().tolist()

    @staticmethod
    def __get_angles(angle_np: pd.DataFrame  # Angels in the itp file
                     ) -> pd.DataFrame:
        """find the angles which involves N and HN3"""
        condition: pd.Series = angle_np['name'].str.contains('-N-') & \
            angle_np['name'].str.contains('HN3')
        unique_angles: pd.DataFrame = \
            angle_np.loc[condition, ['name', 'typ']].drop_duplicates()
        return unique_angles


class UpdateBond:
    """update bonds section by adding new N-HN3 bonds with respective N
    atoms"""

    bonds_updated: pd.DataFrame   # Updated bonds section with new bonds

    def __init__(self,
                 bonds_np: pd.DataFrame,  # Bonds form the itp file for NP
                 hn3: pd.DataFrame,  # New HN3 to add to the atoms
                 atoms: pd.DataFrame  # Updated APTES chains by UpAtom class
                 ) -> None:
        self.bonds_updated = self.update_bonds(bonds_np, hn3, atoms)

    def update_bonds(self,
                     bonds_np: pd.DataFrame,  # Bonds form the itp file for NP
                     hn3: pd.DataFrame,  # New HN3 to add to the atoms
                     atoms: pd.DataFrame  # Updated APTES chains by UpAtom
                     ) -> pd.DataFrame:
        """update the bonds section"""
        hn3_res_atomnr: dict[typing.Any, typing.Any]  # Residue and H atoms ind
        n_res_atomnr: dict[int, int]  # Residue and N atoms index
        hn3_res_atomnr, n_res_atomnr = self.__get_n_index(hn3, atoms)
        new_bonds: pd.DataFrame = self.mk_bonds(hn3_res_atomnr, n_res_atomnr)
        # Concate the bonds and return them
        bonds_updated: pd.DataFrame = self.concate_bonds(bonds_np, new_bonds)
        return bonds_updated

    @staticmethod
    def concate_bonds(bonds_np: pd.DataFrame,  # Bonds in the itp file
                      new_bonds: pd.DataFrame  # New bonds of N-HN3
                      ) -> pd.DataFrame:
        """concate new bonds with old one"""
        bonds_np = bonds_np.reset_index(drop=True)
        new_bonds = new_bonds.reset_index(drop=True)
        return pd.concat([bonds_np, new_bonds], axis=0, ignore_index=True)

    @staticmethod
    def mk_bonds(hn3_res_atomnr: dict[typing.Any, typing.Any],  # HN3 atomnrs
                 n_res_atomnrdict: dict[int, int]  # Residue and N atoms index
                 ) -> pd.DataFrame:
        """make bonds dataframe for N and new HN3"""
        new_bonds = pd.DataFrame({'typ': 1,
                                  'ai': list(n_res_atomnrdict.values()),
                                  'aj': list(hn3_res_atomnr.values()),
                                  'cmt': ';',
                                  'name': 'N-HN3'})
        return new_bonds

    @staticmethod
    def __get_n_index(hn3: pd.DataFrame,  # New NH3 dataframe
                      atoms: pd.DataFrame  # Updated APTES chains by UpAtom
                      ) -> tuple[dict[typing.Any, typing.Any],
                                 dict[int, int]]:
        """return index of the new NH3 atoms"""
        # Getting hn3 atoms index
        hn3_res_atomnr: dict[typing.Any, typing.Any]  # reser and HN3 atomnr
        hn3_res_atomnr = UpdateBond.__get_res_atom_dict(hn3, atoms, 'HN3')
        n_res_atomnr: dict[int, int]  # resnr and N atomnr
        n_res_atomnr = UpdateBond.__get_res_atom_dict(hn3, atoms, 'N')
        return hn3_res_atomnr, n_res_atomnr

    @staticmethod
    def __get_res_atom_dict(hn3: pd.DataFrame,  # New NH3 dataframe
                            atoms: pd.DataFrame,  # Updated APTES chains
                            atom_name: str  # Name of the atom
                            ) -> dict[typing.Any, typing.Any]:
        """nake a dictionary based on the indices for asked atom"""
        hn3_resnr: list[typing.Any]  # Inices are int, but saved strings
        atomnr: list[typing.Any]  # Atom numbers of HN3
        hn3_resnr = hn3['residue_number'].to_list()
        condition: pd.Series = (atoms['atomname'].isin([atom_name])) & \
                               (atoms['resnr'].isin(hn3_resnr))
        atomnr = atoms[condition]['atomnr'].to_list()
        return dict(zip(hn3_resnr, atomnr))


class UpdateAtom:
    """
    Update atom section by adding new HN3 and updating the N, HN1, and
    HN2 atoms.

    This class provides methods to update the atom section of an itp
    file by adding new HN3 atoms and updating charges and atom
    information for N, HN1, and HN2 atoms.

    Attributes:
        atoms_updated (pd.DataFrame): Updated atoms section of the itp
        file.

    Methods:
        __init__(self, atoms_np, hn3, aptes, core)
        update_atoms(self, atoms_np, hn3, aptes, core)
        _calculate_max_indices(atoms, hn3)
        _get_protonated_hn_info(atoms)
        _prepare_hn3_itp_df(hn3, h_n_df, lst_atom)
        _get_atom_info_dict(df_info, key)
        _update_chain_charges(atoms, h_n_df, res_numbers)
        _concatenate_aptes_and_hn3(prepare_hn3, atoms)
        _create_updated_atoms_section(cor_atoms, updated_aptes)
        _perform_charge_check(updates_np)
    """

    atoms_updated: pd.DataFrame  # Updated atoms section of the itp file

    def __init__(self,
                 atoms_np: pd.DataFrame,  # Atoms form the itp file for NP
                 hn3: pd.DataFrame,  # New HN3 to add to the atoms
                 aptes: str,  # Name of the APTES branches
                 core: str  # Name of the CORE residues
                 ) -> None:
        self.atoms_updated = self.update_atoms(atoms_np, hn3, aptes, core)

    def update_atoms(self,
                     atoms_np: pd.DataFrame,  # Atoms form the itp file for NP
                     hn3: pd.DataFrame,  # New HN3 to add to the atoms
                     aptes: str,  # Name of the APTES branches
                     core: str  # Name of the CORE residues
                     ) -> pd.DataFrame:
        """
        Update the atom section of the itp file with new HN3 atoms and
        updated charges.

        Args:
            atoms_np (pd.DataFrame): Atoms from the itp file for the
            nanoparticle.
            hn3 (pd.DataFrame): New HN3 to add to the atoms.
            aptes (str): Name of the APTES branches.
            core (str): Name of the CORE residues.

        Returns:
            pd.DataFrame: Updated atoms section of the itp file.
        """
        # Sanity check of indeces and return the atom index in atoms
        lst_atom: np.int64  # index of the final atoms
        lst_atom = self.__calculate_max_indices(atoms_np, hn3)
        # Get only APT atoms
        atoms: pd.DataFrame = atoms_np[atoms_np['resname'] == aptes]
        # Get COR atoms
        cor_atoms: pd.DataFrame = atoms_np[atoms_np['resname'] == core]
        # Get information for protonated H-N group from the itp file
        h_n_df: pd.DataFrame = self.__get_protonated_hn_info(atoms)
        # Make a dataframe in format of itp for new nh3
        prepare_hn3: pd.DataFrame = \
            self.__prepare_hn3_itp_df(hn3, h_n_df, lst_atom)
        # Update N, HN1, and HN2 charges in the protonated chains
        atoms = self.__update_chain_charges(
            atoms, h_n_df, list(hn3['residue_number']))
        updated_aptes: pd.DataFrame = \
            self.__concat_aptes_and_hn3(prepare_hn3, atoms)
        # get the final atoms section of itp file
        updates_np: pd.DataFrame = \
            self.__create_updated_atoms_section(cor_atoms, updated_aptes)
        self.perform_charge_check(updates_np)
        return updates_np

    def __calculate_max_indices(self,
                                atoms: pd.DataFrame,  # Atoms of the itp file
                                hn3: pd.DataFrame  # New HN3 to add to atoms
                                ) -> np.int64:
        """
        Calculate and return the maximum value of atom and residue
        indices, checking for mismatched residue numbers.

        Args:
            atoms (pd.DataFrame): DataFrame containing atom information
            from the itp file.
            hn3 (pd.DataFrame): New HN3 to add to the atoms.

        Returns:
            np.int64: Maximum value of atom and residue indices.
        """
        atoms['atomnr'] = pd.to_numeric(atoms['atomnr'], errors='coerce')
        atoms['resnr'] = pd.to_numeric(atoms['resnr'], errors='coerce')

        max_atomnr: np.int64 = np.max(atoms['atomnr'])
        lst_atomnr: int = atoms['atomnr'].iloc[-1]
        max_resnr: np.int64 = np.max(atoms['resnr'])
        lst_resnr: int = atoms['resnr'].iloc[-1]

        lst_nh3_res: np.int64  # index of the final residue
        lst_nh3_res = list(hn3['residue_number'])[-1]
        if np.max(np.array([max_resnr, lst_resnr])) != lst_nh3_res:
            print(f'{bcolors.CAUTION}{UpdateAtom.__module__}:\n'
                  '\tThere is a possible mismatch in the new HN3 and initial'
                  ' APTES list or the itp file was updated once before.\n'
                  f'{bcolors.ENDC}')
        return np.max(np.array([max_atomnr, lst_atomnr]))

    @staticmethod
    def __get_protonated_hn_info(atoms: pd.DataFrame,  # Atoms of the itp file
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
            df_tmp[df_tmp['atomname'].isin(['CT', 'N', 'HN1', 'HN2', 'HN3'])]
        if atoms[atoms['atomname'] == 'HN3'].empty:
            sys.exit(f'{bcolors.FAIL}{UpdateAtom.__module__}: \n'
                     '\tError! There is no HN3 in the chosen protonated '
                     f'branch\n{bcolors.ENDC}')
        return df_one

    @staticmethod
    def __prepare_hn3_itp_df(hn3: pd.DataFrame,  # New HN3 atoms, in pdb format
                             h_n_df: pd.DataFrame,  # Info H-N protonated group
                             lst_atom: np.int64  # Last atom index in hn3
                             ) -> pd.DataFrame:
        """make a dataframe in the itp format, for appending to the
        main atom section"""
        columns: list[str]  # Name of the columns
        info: dict[str, typing.Any]  # Info in the row of the df
        columns, info = UpdateAtom.__get_atom_info_dict(h_n_df, 'HN3')
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
    def __update_chain_charges(atoms: pd.DataFrame,  # APTES atoms
                               h_n_df: pd.DataFrame,  # Protonated N-H info
                               res_numbers: list[int]  # Index-chains to prtons
                               ) -> pd.DataFrame:
        """update the N, HN1, and HN2 in the chains which should be
        updated"""
        n_q: float  # Charge of N atom in protonated state
        ct_q: float  # Charge of CT atom in protonated state
        h1_q: float  # Charge of HN1 atom in protonated state
        h2_q: float  # Charge of HN1 atom in protonated state
        n_q = UpdateAtom.__get_atom_info_dict(h_n_df, 'N')[1]['charge']
        ct_q = UpdateAtom.__get_atom_info_dict(h_n_df, 'CT')[1]['charge']
        h1_q = UpdateAtom.__get_atom_info_dict(h_n_df, 'HN1')[1]['charge']
        h2_q = UpdateAtom.__get_atom_info_dict(h_n_df, 'HN2')[1]['charge']
        # Create a condition for selecting rows that need to be updated
        condition = (atoms['atomname'].isin(['CT', 'N', 'HN1', 'HN2'])) & \
                    (atoms['resnr'].isin(res_numbers))

        # Update the 'charge' column for the selected rows
        atoms.loc[condition, 'charge'] = \
            atoms.loc[condition, 'atomname'].map({'N': n_q,
                                                  'CT': ct_q,
                                                  'HN1': h1_q,
                                                  'HN2': h2_q})
        return atoms

    @staticmethod
    def __concat_aptes_and_hn3(prepare_hn3: pd.DataFrame,  # Prepared HN3 df
                               atoms: pd.DataFrame  # Updated chain with charge
                               ) -> pd.DataFrame:
        """concate the dataframes and sort them based on the residue
        indices"""
        updated_atoms: pd.DataFrame = \
            pd.concat([atoms, prepare_hn3], axis=0, ignore_index=False)

        updated_atoms = updated_atoms.sort_values(by=['atomnr', 'resnr'],
                                                  ascending=[True, True])
        return updated_atoms

    @staticmethod
    def __create_updated_atoms_section(cor_atoms: pd.DataFrame,  # Si atoms
                                       updated_aptes: pd.DataFrame  # Chains
                                       ) -> pd.DataFrame:
        """make the final dataframe by appending updated aptes and core
        atoms of the nanoparticle"""
        return pd.concat([cor_atoms, updated_aptes],
                         axis=0, ignore_index=False)

    @staticmethod
    def __get_atom_info_dict(df_info: pd.DataFrame,  # Dataframe to take info
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
    def perform_charge_check(updates_np: pd.DataFrame  # updated Nanoparticle
                             ) -> None:
        """sanity check of the charges in the nanoparticle"""
        # Convert column 'A' to numeric type
        updates_np['charge'] = pd.to_numeric(updates_np['charge'])

        # Get the sum of column 'charge'
        charge_sum = updates_np['charge'].sum()
        if int(charge_sum) != round(charge_sum, 3):
            sys.exit(f'{bcolors.FAIL}{UpdateAtom.__module__}:\n'
                     '\tThe total sum of charges is not a complete number\n'
                     f'\tTotal charge: is `{charge_sum}`\n{bcolors.ENDC}')


class WrapperUpdateItp:
    """An wrapper to update itp files"""
    def __init__(self,
                 param: dict[str, typing.Any],  # All the parameters
                 hn3_dict: dict[str, pd.DataFrame]
                 ) -> None:
        self.updated_itp: dict[str, UpdateItp] = \
            self.update_all_np_itp(param, hn3_dict)

    def update_all_np_itp(self,
                          param: dict[str, typing.Any],  # All the parameters
                          hn3_dict: dict[str, pd.DataFrame]
                          ) -> dict[str, UpdateItp]:
        """
        Update all itp files with HN3 molecules.

        This method iterates through each nanoparticle in the hn3_dict
        and corresponding itp files in param['itp_files'].
        It checks if the nanoparticle name (aptes) is present in the
        itp file name, and if so, extracts the core name.
        Then, it creates an UpdateItp object with the extracted
        information and the HN3 data for the nanoparticle.
        The updated itp object is added to the dictionary updated_itp.

        Args:
            param (dict[str, typing.Any]): All the parameters.
            hn3_dict (dict[str, pd.DataFrame]): Dictionary containing
            HN3 data for each nanoparticle.

        Returns:
            dict[str, UpdateItp]: A dictionary of updated itp files.
        """
        updated_itp: dict[str, UpdateItp] = {}
        for aptes, item in hn3_dict.items():
            for fname_i in param['itp_files']:
                if aptes in fname_i:
                    fname = fname_i
                    itp_i: str = my_tools.drop_string(fname, '.itp')
                    core: str = itp_i.split('_')[1].strip()
                    break
                fname = None
            if fname is not None:
                updated_itp[fname] = UpdateItp(fname, item, aptes, core)
            else:
                sys.exit(f'{bcolors.FAIL}\nThere is a problem in '
                         f'naming the nanoparticles\n{bcolors.ENDC}')
        return updated_itp


class StandAlone:
    """
    Prepare random positions and velocities for some NH3, so I can test
    the script wihtout running whole scripts.
    Input:
        None
    Output:
    """

    new_nh3: pd.DataFrame  # All the new HN3 atoms

    def __init__(self) -> None:
        self.new_nh3 = self.mk_data_standalone()

    def mk_data_standalone(self) -> pd.DataFrame:
        """
        prepare new NH3 positions and velocities for testing the script
        """
        generate: bool = False
        start, end = 1611, 1621

        ind_range: list[int] = self.generate_list(start, end, generate)
        print(f'{bcolors.CAUTION}{self.__module__}, {self.__class__}:\n'
              '\tUsing the module as standalone, make sure items in the '
              f'list:\n\t{ind_range}\n'
              '\tdo not already have HN3 atoms.\n'
              f'{bcolors.ENDC}')
        poistion = \
            {key: self.generate_random_np_array(100) for key in ind_range}
        velocity = {key: self.generate_random_np_array(1) for key in ind_range}
        return self.prepare_hydrogens(poistion, velocity)

    @staticmethod
    def generate_list(start: int,
                      end: int,
                      generate: bool
                      ) -> list[int]:
        """generate a list"""
        if generate:
            return list(range(start, end + 1))
        return [1611, 1613, 1614, 1615, 1617, 1619]

    @staticmethod
    def generate_random_np_array(lim: int  # The limit of the return values
                                 ) -> np.ndarray:
        """
        Generate Random array
        """
        return np.array([np.random.uniform(0, lim),
                         np.random.uniform(0, lim),
                         np.random.uniform(0, lim)
                         ])

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


if __name__ == '__main__':
    import write_itp_file
    write_itp_file.WriteItp(UpdateItp(
                            fname=sys.argv[1],
                            hn3=StandAlone().new_nh3,
                            aptes='APT',
                            core='COR'), fname='APT_COR_updated.itp',
                            log=logger.setup_logger('update_itp.log'))
