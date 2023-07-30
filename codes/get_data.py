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
import gro_to_df as grof
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
    title: str  # Name of the system; if the file is gro
    pbc_box: str  # PBC of the system; if the file is gro

    def __init__(self,
                 fname: str,  # Name of the pdb file
                 log: logger.logging.Logger
                 ) -> None:
        self.param = param.ReadParam(log=log).param
        self.atoms = self.__get_data(fname, log)
        self.residues_atoms = self.__get_atoms()
        # All the unprtonated aptes to be protonated:
        self.unproton_aptes, self.unprot_aptes_ind = self.process_data(log)
        self.np_diameter = self.__get_np_size()
        self.__write_msg(log)
        self.info_msg = ''  # Empety the msg

    def __get_data(self,
                   fname: str,  # Name of the pdb file
                   log: logger.logging.Logger
                   ) -> typing.Any:
        """select which datafile to work with"""
        if self.param['FILE'] == 'PDB':
            atoms = pdbf.Pdb(fname, log).atoms
        elif self.param['FILE'] == 'GRO':
            gro = grof.ReadGro(fname, log)
            atoms = gro.gro_data
            self.title = gro.title
            self.pbc_box = gro.pbc_box
        return atoms

    def process_data(self,
                     log: logger.logging.Logger
                     ) -> tuple[np.ndarray, list[int]]:
        """Check and find the unprotonated APTES group that has N at
        the interface.

        Parameters:
            log (Logger): A logging.Logger instance for logging
            information.

        Returns:
            Tuple[np.ndarray, List[int]]: A tuple containing two elem-
            ents:
            - A numpy array containing all atoms in the chains of the
              unprotonated APTES residues.
            - A list of integers representing the indices of unproton-
              ated APTES residues to be protonated.
        """
        # Get the water surface
        water_surface = \
            pdb_surf.GetSurface(self.residues_atoms, log, write_debug=False)

        # Get the z-axis range of the water surface interface
        zrange: tuple[float, float] = self.__get_interface_range(water_surface)

        # Get the indices of all the APTES residues at the sol phase interface
        sol_phase_aptes: list[int] = self.__get_aptes_indices(zrange)

        # Get the indices of the unprotonated APTES residues to be protonated
        unprot_aptes_ind: list[int] = self.__get_unprto_chain(sol_phase_aptes)

        # Log the number of chains to be protonated
        self.info_msg += \
            f'\tNumber of chains to be protonated: {len(unprot_aptes_ind)}\n'

        # Return a tuple containing the DataFrame of unprotonated APTES
        # chains and the list of their indices
        return self.get_aptes_unproto(unprot_aptes_ind), unprot_aptes_ind

    def get_aptes_unproto(self,
                          unprot_aptes_ind: list[int]  # Index of the APTES
                          ) -> pd.DataFrame:
        """Get all atoms in the chains of the unprotonated APTES.

        Parameters:
            unprot_aptes_ind (List[int]): A list of integers representing
            the indices of unprotonated APTES residues.

        Returns:
            pd.DataFrame: DataFrame containing all atoms in the chains
            of the unprotonated APTES residues.
        """
        # Access the DataFrame containing APTES atom data
        df_apt: pd.DataFrame = self.residues_atoms['APT']

        # Filter the DataFrame to get all atoms in the chains of\
        # unprotonated APTES
        unprotonated_aptes_df = \
            df_apt[df_apt['residue_number'].isin(unprot_aptes_ind)]

        # Return the DataFrame containing all atoms in the chains of
        # the unprotonated APTES
        return unprotonated_aptes_df

    def __get_unprto_chain(self,
                           sol_phase_aptes: list[int]  # Indices of APTES
                           ) -> list[int]:
        """Find all the chains at the interface that require protonation.

        Parameters:
            sol_phase_aptes (List[int]): Indices of APTES residues.

        Returns:
            List[int]: A list of integers representing the indices of
                       APTES
            residues that require protonation.
        """
        # Get the DataFrame for APTES atoms
        df_apt: pd.DataFrame = self.residues_atoms['APT']

        # Initialize an empty list to store unprotonated APTES indices
        unprotonated_aptes: list[int] = []

        # Split the sol_phase_aptes list into chunks for parallel processing
        num_processes: int = multip.cpu_count() // 2
        chunk_size: int = len(sol_phase_aptes) // num_processes
        chunks = [sol_phase_aptes[i:i + chunk_size] for i in
                  range(0, len(sol_phase_aptes), chunk_size)]

        # Create a Pool of processes
        with multip.Pool(processes=num_processes) as pool:
            # Process the chunks in parallel using the process_chunk function
            results = pool.starmap(self.process_chunk,
                                   [(chunk, df_apt) for chunk in chunks])

        # Combine the results from each process
        for result in results:
            unprotonated_aptes.extend(result)

        # Release memory by deleting the DataFrame
        del df_apt

        # Return the list of unprotonated APTES indices
        return unprotonated_aptes

    @staticmethod
    def process_chunk(chunk: np.ndarray,  # Chunk of a APTES indices
                      df_apt: pd.DataFrame  # For the APTES at the interface
                      ) -> list[int]:
        """
        Process a chunk of APTES residues to find unprotonated chains.

        Parameters:
            chunk (np.ndarray): A chunk of APTES indices to process.
            df_apt (pd.DataFrame): DataFrame containing APTES atom data.

        Returns:
            List[int]: A list of integers representing the indices of
            unprotonated APTES residues within the chunk.
        """
        # Initialize an empty list to store unprotonated APTES indices
        # in the chunk
        unprotonated_aptes_chunk: list[int] = []

        # Iterate over the APTES indices in the chunk
        for aptes_index in chunk:
            # Filter the DataFrame for the current APTES index
            df_i = df_apt[df_apt['residue_number'] == aptes_index]
            # Check if 'HN3' is present in 'atom_name' for the current
            # APTES residue
            if df_i[df_i['atom_name'].isin(['HN3'])].empty:
                # If 'HN3' is not present, add the index to the list
                # of unprotonated APTES
                unprotonated_aptes_chunk.append(aptes_index)

        # Return the list of unprotonated APTES indices in the chunk
        return unprotonated_aptes_chunk

    def __get_aptes_indices(self,
                            zrange: tuple[float, float]  # Bound of interface
                            ) -> list[int]:
        """
        Get the indices of APTES residues within the specified z-axis
        range.

        Parameters:
            zrange (Tuple[float, float]): A tuple containing the lower
            and upper bounds of the z-axis range.

        Returns:
            List[int]: A list of integers representing the indices of
            APTES residues that lie within the specified z-axis range.
        """
        if not isinstance(zrange, tuple) or len(zrange) != 2:
            raise ValueError(
                "zrange must be a tuple containing two float values.")

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
        """Find all the APTES residues at the interface.

        Parameters:
            water_surface (Any): The result of pdb_surf.GetSurface.

        Returns:
            Tuple[float, float]: A tuple containing the lower and upper
            bounds of the z-axis range that defines the interface.
        """
        z_range: tuple[float, float]

        if self.param['READ'] == 'False':
            self.info_msg += '\tInterface data is read from update_param\n'
            # Interface is set with reference to the NP COM
            interface_z = self.param['INTERFACE']
            interface_w = self.param['INTERFACE_WIDTH']
            aptes_com = self.param['NP_ZLOC']
        elif self.param['READ'] == 'True':
            # Interface is calculated directly
            self.info_msg += \
                '\tInterface data is selected from the input file\n'
            interface_z = water_surface.interface_z
            interface_w = water_surface.interface_std * 2
            aptes_com = 0

        z_range = self.__interface_range(interface_z, interface_w, aptes_com)
        return z_range

    def __interface_range(self,
                          interface_z: float,  # Location of interface
                          interface_w: float,  # Width of interface
                          aptes_com: float,  # COM of center of mass
                          ) -> tuple[float, float]:
        """Set the interface range.

        Parameters:
            interface_z (float): Location of the interface.
            interface_w (float): Width of the interface.
            aptes_com (float): COM (Center of Mass) of the center of
            mass.

        Returns:
            Tuple[float, float]: A tuple containing the lower and upper
            bounds of the z-axis range that defines the interface.
        """
        if self.param['LINE'] == 'WITHIN':
            self.info_msg += \
                '\tOnly checks APTES in the width of the interface\n'
            z_range = (interface_z - interface_w/2 + aptes_com,
                       interface_z + interface_w/2 + aptes_com)
        elif self.param['LINE'] == 'INTERFACE':
            self.info_msg += \
                '\tChecks APTES under the interface (average value)\n'
            z_range = (0, interface_z + aptes_com)
        elif self.param['LINE'] == 'LOWERBOUND':
            self.info_msg += \
                '\tChecks APTES under the interface - standard deviation\n'
            z_range = (0, interface_z - interface_w/2 + aptes_com)
        elif self.param['LINE'] == 'UPPERBOUND':
            self.info_msg += \
                '\tChecks APTES under the interface + standard deviation\n'
            z_range = (0, interface_z + interface_w/2 + aptes_com)
        else:
            sys.exit(f'{self.__module__}:\n\tError! '
                     f'INTERFACE selection failed')

        return z_range

    def __get_atoms(self) -> dict[str, pd.DataFrame]:
        """Get all the atoms for each residue.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing pandas
            DataFrames for each residue's atoms.
        """
        # Get the names of all residues
        residues: list[str] = self.__get_residues_names()

        # Get the atoms for each residue and store them in a dictionary
        residues_atoms: dict[str, pd.DataFrame] = \
            self.__get_residues_atoms(residues)

        # Get the atoms in the bounding box enclosing the NP and add
        # them to the dictionary
        residues_atoms['box'] = self.__get_np_box(residues_atoms)

        return residues_atoms

    def __get_residues_atoms(self,
                             residues: list[str]  # Name of the residues
                             ) -> dict[str, pd.DataFrame]:
        """
        Return a dictionary of all the residues with their atoms
        information.

        Parameters:
            residues (list[str]): Names of the residues.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing pandas
            DataFrames for each residue's atoms.
        """
        residues_atoms: dict[str, pd.DataFrame] = {}  # All the atoms data
        for res in residues:
            # Filter the atoms DataFrame to get atoms belonging to each
            # residue and store them in the dictionary.
            residues_atoms[res] = self.atoms[self.atoms['residue_name'] == res]
        return residues_atoms

    def __get_np_box(self,
                     residues_atoms: dict[str, pd.DataFrame]
                     ) -> pd.DataFrame:
        """
        Get area around NP and obtain a box of all the residues in that
        box.

        Parameters:
            residues_atoms (Dict[str, pd.DataFrame]): A dictionary
            containing pandas DataFrames for each residue's atoms.

        Returns:
            pd.DataFrame: A DataFrame containing atoms inside the
            bounding box around the NP.
        """
        xrange: tuple[float, float]  # Range of NP in x direction
        yrange: tuple[float, float]  # Range of NP in y direction
        zrange: tuple[float, float]  # Range of NP in z direction
        # Get the x, y, and z ranges of the NP using the __get_np_range method.
        xrange, yrange, zrange = self.__get_np_range(residues_atoms['APT'])
        # Get the atoms inside the bounding box using the __get_inside_box
        # method.
        return self.__get_inside_box(xrange, yrange, zrange)

    def __get_inside_box(self,
                         xrange: tuple[float, float],  # Range of NP in x
                         yrange: tuple[float, float],  # Range of NP in y
                         zrange: tuple[float, float]  # Range of NP in z
                         ) -> pd.DataFrame:
        """
        Get atoms inside the box defined by the given x, y, and z anges.

        Parameters:
            xrange (tuple[float, float]): Range of the NP in the x
                                          direction.
            yrange (tuple[float, float]): Range of the NP in the y
                                          direction.
            zrange (tuple[float, float]): Range of the NP in the z
                                          direction.

        Returns:
            pd.DataFrame: A DataFrame containing atoms inside the
            specified box.
        """
        # Increase the box size in each direction by this value.
        epsilon: float = 3
        # Filter the atoms DataFrame to get atoms within the specified
        # box range and return them.
        atoms_box = self.atoms[
            (self.atoms['x'] >= xrange[0] - epsilon) &
            (self.atoms['x'] <= xrange[1] + epsilon) &
            (self.atoms['y'] >= yrange[0] - epsilon) &
            (self.atoms['y'] <= yrange[1] + epsilon) &
            (self.atoms['z'] >= zrange[0] - epsilon) &
            (self.atoms['z'] <= zrange[1] + epsilon)
        ]
        return atoms_box

    @staticmethod
    def __get_np_range(aptes_atoms: pd.DataFrame  # All APTES atoms
                       ) -> tuple[tuple[float, float],
                                  tuple[float, float],
                                  tuple[float, float]]:
        """
        Get the xyz range of the NP.

        Parameters:
            aptes_atoms (pd.DataFrame): All APTES atoms DataFrame.

        Returns:
            Tuple[Tuple[float, float], Tuple[float, float],
            Tuple[float, float]]:
            A tuple containing the x, y, and z ranges of the NP.
        """
        xrange: tuple[float, float] = \
            (aptes_atoms['x'].min(), (aptes_atoms['x'].max()))
        yrange: tuple[float, float] = \
            (aptes_atoms['y'].min(), (aptes_atoms['y'].max()))
        zrange: tuple[float, float] = \
            (aptes_atoms['z'].min(), (aptes_atoms['z'].max()))
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
        """
        Get the list of the residues in the system.

        Returns:
            List[str]: A list containing the names of the residues in
            the system.
        """
        # Get the unique residue names from the 'residue_name' column
        # in the atoms DataFrame.
        residues: list[str] = list(set(self.atoms['residue_name']))
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
