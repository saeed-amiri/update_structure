"""The purpose of this script is to divide the data file and extract
the relevant section of data. It creates separate data frames for
different residues or groups of residues. The data is accessed through
pdb_todf.py."""


import sys
import multiprocessing as multip
import typing
import numpy as np
import pandas as pd
import pdb_to_df as pdbf
import read_param as param
import get_interface as pdb_surf
import logger
from colors_text import TextColor as bcolors


class ProcessData:
    """process dataframe of the structure and plit them"""

    info_msg: str = 'Message:\n'  # Message to pass for logging and writing
    atoms: pd.DataFrame  # All atoms dataframe
    param: dict[str, float]  # All the parameters from input file
    residues_atoms: dict[str, pd.DataFrame]  # Atoms info for each residue
    unproton_aptes: pd.DataFrame  # APTES which should be protonated
    unprot_aptes_ind: list[int]  # Index of APTES which should be protonated
    np_diameter: np.float64  # Diameter of NP, based on APTES positions

    def __init__(self,
                 fname: str,  # Name of the pdb file
                 log: logger.logging.Logger
                 ) -> None:
        self.param = param.ReadParam(log=log).param
        self.atoms = pdbf.Pdb(fname, log).atoms
        self.residues_atoms = self.__get_atoms()
        # All the unprtonated aptes to be protonated:
        self.unproton_aptes, self.unprot_aptes_ind = self.process_data(log)
        self.np_diameter = self.__get_np_size()
        self.__write_msg(log)
        self.info_msg = ''  # Empety the msg

    def process_data(self,
                     log: logger.logging.Logger
                     ) -> tuple[np.ndarray, list[int]]:
        """check and finds the unprotonated aptes group which has N at
        interface"""
        # Get the water surface
        water_surface = pdb_surf.GetSurface(self.residues_atoms,
                                            log,
                                            write_debug=False)
        zrange: tuple[float, float]  # Lower and upper bound of interface
        zrange = self.__get_interface_range(water_surface)
        sol_phase_aptes: list[int]  # Indices of all the APTES at sol phase
        sol_phase_aptes = self.__get_aptes_indices(zrange)
        unprot_aptes_ind: list[int]  # Index of the APTES to be protonated
        unprot_aptes_ind = self.__get_unprto_chain(sol_phase_aptes)
        self.info_msg += '\tNumber of chains to be protonated: '
        self.info_msg += f'{len(unprot_aptes_ind)}\n'
        return self.get_aptes_unproto(unprot_aptes_ind), unprot_aptes_ind

    def get_aptes_unproto(self,
                          unprot_aptes_ind: list[int]  # Index of the APTES
                          ) -> pd.DataFrame:
        """get all atoms in the chains of the unprotonated APTES"""
        df_apt: pd.DataFrame = self.residues_atoms['APT']
        return df_apt[df_apt['residue_number'].isin(unprot_aptes_ind)]

    def __get_unprto_chain(self,
                           sol_phase_aptes: list[int]  # Indices of APTES
                           ) -> list[int]:
        """find all the chains at the intrface"""
        df_apt: pd.DataFrame = self.residues_atoms['APT']
        unprotonated_aptes: list[int] = []
        # Split the sol_phase_aptes list into chunks
        num_processes: int = multip.cpu_count() // 2
        chunk_size: int = len(sol_phase_aptes) // num_processes
        chunks = [sol_phase_aptes[i:i+chunk_size] for i in
                  range(0, len(sol_phase_aptes), chunk_size)]
        # Create a Pool of processes
        with multip.Pool(processes=num_processes) as pool:
            # Process the chunks in parallel
            results = pool.starmap(self.process_chunk,
                                   [(chunk, df_apt) for chunk in chunks])
        # Combine the results
        for result in results:
            unprotonated_aptes.extend(result)
        del df_apt
        return unprotonated_aptes

    @staticmethod
    def process_chunk(chunk: np.ndarray,  # Chunk of a APTES indices
                      df_apt: pd.DataFrame  # For the APTES at the interface
                      ) -> list[int]:
        """get index of unprotonated aptes"""
        unprotonated_aptes_chunk: list[int] = []  # Index of unprotonated APTES
        for ind in chunk:
            df_i = df_apt[df_apt['residue_number'] == ind]
            # Check if 'NH3' is present in 'atom_name'
            if df_i[df_i['atom_name'].isin(['HN3'])].empty:
                unprotonated_aptes_chunk.append(ind)
        return unprotonated_aptes_chunk

    def __get_aptes_indices(self,
                            zrange: tuple[float, float]  # Bound of interface
                            ) -> list[int]:
        """get the index of all the aptes at the interface range"""
        # Filter the DataFrame based on the specified conditions
        df_apt = self.residues_atoms['APT']
        filtered_df = df_apt[(df_apt['atom_name'] == 'N') &
                             (df_apt['z'].between(zrange[0], zrange[1]))]

        # Get the 'residue_number' values for the filtered atoms
        del df_apt
        return filtered_df['residue_number'].values

    def __get_interface_range(self,
                              water_surface: typing.Any  # pdb_surf.GetSurface
                              ) -> tuple[float, float]:
        """find all the aptes at interface."""
        z_range: tuple[float, float]
        if self.param['PDB'] == 'False':
            self.info_msg += '\tInterface data is read from update_param\n'
            # Interface is set with reference to the NP COM
            interface_z = self.param['INTERFACE']
            interface_w = self.param['INTERFACE_WIDTH']
            aptes_com = self.param['NP_ZLOC']
        elif self.param['PDB'] == 'True':
            # Interface is calculated directly
            self.info_msg += '\tInterface data is selcected from pdb file\n'
            interface_z = water_surface.interface_z
            interface_w = water_surface.interface_std * 4
            aptes_com = 0
        z_range = self.__interface_range(interface_z,
                                         interface_w,
                                         aptes_com)
        return z_range

    def __interface_range(self,
                          interface_z: float,  # Location of interface
                          interface_w: float,  # Width of interface
                          aptes_com: float,  # COM of center of mass
                          ) -> tuple[float, float]:
        """set the interface range"""
        if self.param['LINE'] == 'WITHIN':
            self.info_msg += \
                '\tOnly checks APTES in the width of interface\n'
            z_range = (interface_z-interface_w/2 + aptes_com,
                       interface_z+interface_w/2 + aptes_com)
        elif self.param['LINE'] == 'INTERFACE':
            self.info_msg += '\tChecks APTES under interface (average value)\n'
            z_range = (0, interface_z + aptes_com)
        elif self.param['LINE'] == 'LOWERBOUND':
            self.info_msg += \
                '\tChecks APTES under interface - standard diviation\n'
            z_range = (0, interface_z - interface_w/2 + aptes_com)
        elif self.param['LINE'] == 'UPPERBOUND':
            self.info_msg += \
                '\tChecks APTES under interface + standard diviation\n'
            z_range = (0, interface_z + interface_w/2 + aptes_com)
        else:
            sys.exit(f'{self.__module__}:\n'
                     '\tError! INTERFACE selection failed')
        return z_range

    def __get_atoms(self) -> dict[str, pd.DataFrame]:
        """get all the atoms for each residue"""
        residues: list[str] = self.__get_residues_names()
        residues_atoms: dict[str, pd.DataFrame] = \
            self.__get_residues_atoms(residues)
        residues_atoms['box'] = self.__get_np_box(residues_atoms)
        return residues_atoms

    def __get_residues_atoms(self,
                             residues: list[str]  # Name of the residues
                             ) -> dict[str, pd.DataFrame]:
        """return a dictionary of all the residues with thier atoms
        information"""
        residues_atoms: dict[str, pd.DataFrame] = {}  # All the atoms data
        for res in residues:
            residues_atoms[res] = self.atoms[self.atoms['residue_name'] == res]
        return residues_atoms

    def __get_np_box(self,
                     residues_atoms: dict[str, pd.DataFrame]
                     ) -> pd.DataFrame:
        """get area around NP and get a box of all the residue in that
        box"""
        xrange: tuple[float, float]  # Range of NP in x direction
        yrange: tuple[float, float]  # Range of NP in y direction
        zrange: tuple[float, float]  # Range of NP in z direction
        xrange, yrange, zrange = self.__get_np_range(residues_atoms['APT'])
        return self.__get_inside_box(xrange, yrange, zrange)

    def __get_inside_box(self,
                         xrange: tuple[float, float],  # Range of NP in x
                         yrange: tuple[float, float],  # Range of NP in y
                         zrange: tuple[float, float]  # Range of NP in z
                         ) -> pd.DataFrame:
        """get atoms inside the box"""
        epsilon: float = 3  # increase the box in each direction
        atoms_box: pd.DataFrame  # Atoms inside the box
        atoms_box = self.atoms[
                               (self.atoms['x'] >= xrange[0]-epsilon) &
                               (self.atoms['x'] <= xrange[1]+epsilon) &
                               (self.atoms['y'] >= yrange[0]-epsilon) &
                               (self.atoms['y'] <= yrange[1]+epsilon) &
                               (self.atoms['z'] >= zrange[0]-epsilon) &
                               (self.atoms['z'] <= zrange[1]+epsilon)
                               ]
        return atoms_box

    @staticmethod
    def __get_np_range(aptes_atoms: pd.DataFrame  # All APTES atoms
                       ) -> tuple[tuple[float, float],
                                  tuple[float, float],
                                  tuple[float, float]]:
        """get the xyz range of NP"""
        xrange: tuple[float, float] = \
            (aptes_atoms['x'].min(), aptes_atoms['x'].max())
        yrange: tuple[float, float] = \
            (aptes_atoms['y'].min(), aptes_atoms['y'].max())
        zrange: tuple[float, float] = \
            (aptes_atoms['z'].min(), aptes_atoms['z'].max())
        return xrange, yrange, zrange

    def __get_np_size(self) -> np.float64:
        """get the maximum radius of NP, since APTES are most outward,
        here only looking at APTES residues"""
        aptes_atoms: pd.DataFrame = self.residues_atoms['APT']
        diameter: list[float] = []  # To save the diameters in each direction
        for xyz in ['x', 'y', 'z']:
            diameter.append(aptes_atoms[xyz].max() - aptes_atoms[xyz].min())
        return np.max(diameter)

    def __get_residues_names(self) -> list[str]:
        """get the list of the residues in the system"""
        residues: list[str]   # Name of the residues
        residues = list(set(self.atoms['residue_name']))
        return residues

    def __write_msg(self,
                    log: logger.logging.Logger
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ProcessData.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    data = ProcessData(sys.argv[1], log=logger.setup_logger('update.log'))
