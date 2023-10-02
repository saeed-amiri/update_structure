"""
ProcessData Class
-----------------

The ProcessData class is designed to efficiently process data from a
pdb or gro file and extract relevant sections of data for different
residues or residue groups. It creates separate data frames for diffe-
rent residues or groups of residues and performs various operations to
analyze and manipulate the data. The class utilizes multiprocessing to
improve performance and is capable of finding unprotonated APTES groups
at the interface, calculating the diameter of nanoparticles (NPs) based
on APTES positions, and setting the interface range.

Attributes:
    atoms (pd.DataFrame): A pandas DataFrame containing data for all
    atoms in the system.
    param (dict[str, float]): A dictionary containing various paramet-
    ers from an input file.
    residues_atoms (dict[str, pd.DataFrame]): A dictionary containing
    pandas DataFrames for each residue's atoms.
    unproton_aptes (dict[str, pd.DataFrame]): A dictionary containing
    pandas DataFrames for each residue's atoms in the chains of the
    unprotonated APTES residues.
    unprot_aptes_ind (list[int]): A list of integers representing the
    indices of unprotonated APTES residues to be protonated.
    np_diameter (np.float64): The maximum radius of the nanoparticle
    (NP) based on APTES positions.
    title (str): The name of the system; applicable if the file is in
    gro format.
    pbc_box (str): The periodic boundary condition of the system;
    applicable if the file is in gro format.

Methods:
    __init__(fname: str, log: logger.logging.Logger) -> None:
        Initialize the ProcessData object.

    find_unprotonated_aptes(log: logger.logging.Logger) ->
                            tuple[dict[str, np.ndarray], list[int]]:
        Check and find the unprotonated APTES groups that have N at
        the interface.

    get_aptes_unproto(unprot_aptes_ind: dict[str, list[int]]) ->
                      dict[str, pd.DataFrame]:
        Get all atoms in the chains of the unprotonated APTES.

    find_unprotonated_aptes_chains(sol_phase_aptes:
                                   dict[str, list[int]]) ->
                                   dict[str, list[int]]:
        Find all the chains at the interface that require protonation.

    calculate_maximum_np_radius() -> np.float64:
        Calculate the maximum radius of the nanoparticle (NP) based on
        APTES positions.

    calculate_np_xyz_range(aptes_atoms: pd.DataFrame) ->
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        Calculate the x, y, and z ranges of the nanoparticle (NP)
        based on APTES coordinates.

    find_interface_z_range(water_surface: typing.Any) ->
                           tuple[float, float]:
        Find all the APTES residues at the interface.

    calculate_interface_z_range(interface_z: float, interface_w: float,
    aptes_com: float) -> tuple[float, float]:
        Set the interface range.

    get_unique_residue_names() -> list[str]:
        Get the list of unique residue names in the system.

Private Methods:
    _get_data(fname: str, log: logger.logging.Logger) -> pd.DataFrame:
        Select which data file to work with and load the atoms data.

    _get_atoms() -> dict[str, pd.DataFrame]:
        Get all the atoms for each residue.

    _get_residues_atoms(residues: list[str]) -> dict[str, pd.DataFrame]:
        Return a dictionary of all the residues with their atoms
        information.

    _get_inside_box(xrange: tuple[float, float], yrange: tuple[float,
    float], zrange: tuple[float, float]) -> pd.DataFrame:
        Get atoms inside the box defined by the given x, y, and z ranges.

    _write_msg(log: logger.logging.Logger) -> None:
        Write and log messages.

Note:
    - This class is intended to be used with pdb and gro files.
    - The class uses multiprocessing to improve performance during
    certain operations.
    - The script contains various methods to analyze and manipulate
    data related to residues and atoms in the system.
    - The class provides methods to find unprotonated APTES groups at
    the interface and calculate the diameter of NPs based on APTES
    positions.
    - The 'param' attribute is populated with parameters from an input
    file to control various aspects of the analysis.
    - It is recommended to initialize the class using the '__init__'
    method with the filename of the pdb or gro file and a logger object
    for logging messages.

Example:
    data = \
        ProcessData("example.pdb", log=logger.setup_logger("update.log"))
"""


import sys
import multiprocessing as multip
import typing
import numpy as np
import pandas as pd
import logger
import cpuconfig
import gro_to_df as grof
import pdb_to_df as pdbf
import read_param as param
import get_interface as pdb_surf
from colors_text import TextColor as bcolors
if typing.TYPE_CHECKING:
    from get_interface import WrapperGetSurface


class ProcessData:
    """
    Process the data and extract relevant sections of data for differ-
    ent residues or residue groups.

    The purpose of this script is to divide the data file and extract
    the relevant section of data. It creates separate data frames for
    different residues or groups of residues. The data is accessed th-
    rough pdb_todf.py.
    """

    info_msg: str = 'Message:\n'  # Message to pass for logging and writing
    atoms: pd.DataFrame  # All atoms dataframe
    param: dict[str, typing.Any]  # All the parameters from input file
    residues_atoms: dict[str, pd.DataFrame]  # Atoms info for each residue
    unproton_aptes: dict[str, pd.DataFrame]  # APTES which should be protonated
    unprot_aptes_ind: dict[str, list[int]]  # Index of APTES to be protonated
    np_diameter: np.float64  # Diameter of NP, based on APTES positions
    title: str  # Name of the system; if the file is gro
    pbc_box: str  # PBC of the system; if the file is gro

    def __init__(self,
                 fname: str,  # Name of the pdb file
                 log: logger.logging.Logger
                 ) -> None:
        """
        Initialize the ProcessData object.

        Parameters:
            fname (str): Name of the pdb file.
            log (Logger): The logger object to log messages.
        """
        # Read parameters from the param file
        self.param = param.ReadParam(log=log).param

        # Get the number of cores on the host
        self.core_nr: int = self.get_nr_cores(log)

        # Load atoms data from the specified file
        self.atoms = self.__get_data(fname, log)

        # Extract atoms data for each residue and store them in a dictionary
        self.residues_atoms = self.__get_atoms()

        # Find unprotonated APTES residues at the interface
        unproton_aptes, unprot_aptes_ind = \
            self.find_unprotonated_aptes(log)
        if self.param['NUMAPTES'] != -1:
            self.info_msg += ('\tThe number of unprotonated aptes is set'
                              f' to {int(self.param["NUMAPTES"])}\n')
        self.unproton_aptes, self.unprot_aptes_ind = \
            self.select_lowest_aptes(unproton_aptes)
        # Get the diameter of the NP
        self.np_diameter = self.calculate_maximum_np_radius()

        # Write and log the initial message
        self.__write_msg(log)

        # Empty the message
        self.info_msg = ''

    def get_nr_cores(self,
                     log: logger.logging.Logger
                     ) -> int:
        """get the number of the available cores"""
        cpu_info = cpuconfig.ConfigCpuNr(log=log)
        return cpu_info.core_nr

    def __get_data(self,
                   fname: str,  # Name of the pdb file
                   log: logger.logging.Logger
                   ) -> typing.Any:
        """
        Select which datafile to work with and load the atoms data.

        Parameters:
            fname (str): Name of the pdb file.
            log (Logger): The logger object to log messages.

        Returns:
            typing.Any: The atoms data.
        """
        if self.param['FILE'] == 'PDB':
            # Load atoms data from PDB file
            atoms = pdbf.Pdb(fname, log).atoms
        elif self.param['FILE'] == 'GRO':
            # Load atoms data from GRO file
            gro = grof.ReadGro(fname, log)
            atoms = gro.gro_data
            self.title = gro.title
            self.pbc_box = gro.pbc_box
        return atoms

    def find_unprotonated_aptes(self,
                                log: logger.logging.Logger
                                ) -> tuple[dict[str, np.ndarray],
                                           dict[str, list[int]]]:
        """Check and find the unprotonated APTES group that has N at
        the interface.

        Parameters:
            log (Logger): A logging.Logger instance for logging
            information.

        Returns:
            Tuple[dict[str, np.ndarray], List[int]]: A tuple containing
            two elements:
            - A numpy array containing all atoms in the chains of the
              unprotonated APTES residues.
            - A list of integers representing the indices of unproton-
              ated APTES residues to be protonated.
        """
        # Get the water surface
        water_surface = pdb_surf.WrapperGetSurface(self.residues_atoms,
                                                   log,
                                                   self.param)

        # Get the z-axis range of the water surface interface
        zrange: tuple[float, float] = \
            self.find_interface_z_range(water_surface)

        # Get the indices of all the APTES residues at the sol phase interface
        sol_phase_aptes: dict[str, list[int]] = \
            self.__get_aptes_indices(zrange)

        # Get the indices of the unprotonated APTES residues to be protonated
        unprot_aptes_ind: dict[str, list[int]] = \
            self.find_unprotonated_aptes_chains(sol_phase_aptes)

        # Return a tuple containing the DataFrame of unprotonated APTES
        # chains and the list of their indices
        return self.get_aptes_unproto(unprot_aptes_ind), unprot_aptes_ind

    def select_lowest_aptes(self,
                            unproton_aptes: dict[str, pd.DataFrame]
                            ) -> tuple[dict[str, pd.DataFrame], ...]:
        """if the numbers of found aptes is more then NUMAPTES
        chose the lowest one
        """
        lowest_amino: dict[str, pd.DataFrame] = {}
        lowest_amino_ind: dict[str, int] = {}
        if (aptes_nr := int(self.param['NUMAPTES'])) != -1:
            for apt, item in unproton_aptes.items():
                if len(item) > aptes_nr:
                    lowest_amino[apt], lowest_amino_ind[apt] = \
                        self.find_lowest_amino_groups(item, aptes_nr)
        return unproton_aptes, lowest_amino_ind

    @staticmethod
    def find_lowest_amino_groups(unproton_aptes: pd.DataFrame,
                                 aptes_nr: int
                                 ) -> pd.DataFrame:
        """find lowest amino groups"""
        df_c: pd.DataFrame = unproton_aptes[unproton_aptes['atom_name'] == "N"]
        lowest_amino_index: list[int] = \
            df_c.nsmallest(aptes_nr, 'z')['residue_number']

        return unproton_aptes[
            unproton_aptes['residue_number'].isin(lowest_amino_index)], \
            list(lowest_amino_index)

    def get_aptes_unproto(self,
                          unprot_aptes_ind: dict[str, list[int]]  # Aptes index
                          ) -> tuple[str, dict[str, pd.DataFrame]]:
        """Get all atoms in the chains of the unprotonated APTES.

        Parameters:
            unprot_aptes_ind (dict[str, list[int]]): A list of integers
            representing the indices of unprotonated APTES residues.

        Returns:
            dict[str, pd.DataFrame]: DataFrame containing all atoms
            in the chains of the unprotonated APTES residues.
        """
        unprotonated_aptes_df_dict: dict[str, pd.DataFrame] = {}
        for aptes, item in unprot_aptes_ind.items():
            # Access the DataFrame containing APTES atom data
            df_apt: pd.DataFrame = self.residues_atoms[aptes]
            # Filter the DataFrame to get all atoms in the chains of\
            # unprotonated APTES
            unprotonated_aptes_df_dict[aptes] = \
                df_apt[df_apt['residue_number'].isin(item)]
        return unprotonated_aptes_df_dict

    def find_unprotonated_aptes_chains(self,
                                       sol_phase_aptes: dict[str, list[int]]
                                       ) -> dict[str, list[int]]:
        """
        Find all the chains at the interface that require protonation.

        Parameters:
            sol_phase_aptes dict[str, list[int]]): Indices of APTES
            residues.

        Returns:
            dict[str, list[int]]: A list of integers representing the
                                  indices of APTES
            residues that require protonation.
        """
        # Initialize an empty list to store unprotonated APTES indices
        unprotonated_aptes: dict[str, list[int]] = {}
        for aptes, item in sol_phase_aptes.items():
            aptes_list: list[int] = []
            # Get the DataFrame for APTES atoms
            df_apt: pd.DataFrame = self.residues_atoms[aptes]

            # Split the sol_phase_aptes into chunks for parallel processing
            chunk_size: int = len(item) // self.core_nr
            chunks = [item[i:i + chunk_size] for i in
                      range(0, len(item), chunk_size)]

            # Create a Pool of processes
            with multip.Pool(processes=self.core_nr) as pool:
                # Process chunks in parallel using the process_chunk function
                results = pool.starmap(self.process_chunk,
                                       [(chunk, df_apt) for chunk in chunks])

            # Release memory by deleting the DataFrame
            del df_apt
            # Combine the results from each process
            for result in results:
                aptes_list.extend(result)
            self.info_msg += ('\tThe number of unprotonated aptes in water: '
                              f'`{aptes}` is {len(aptes_list)}\n')
            unprotonated_aptes[aptes] = aptes_list

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
                            ) -> dict[str, list[int]]:
        """
        Get the indices of APTES residues within the specified z-axis
        range.

        Parameters:
            zrange (Tuple[float, float]): A tuple containing the lower
            and upper bounds of the z-axis range.

        Returns:
            dict[str, list[int]]: A dict of lists of integers represe-
            nting the indices of APTES residues that lie within the
            specified z-axis range.
        """
        if not isinstance(zrange, tuple) or len(zrange) != 2:
            raise ValueError(
                "zrange must be a tuple containing two float values.")

        # Filter the DataFrame based on the specified conditions
        aptes_index_dict: dict[str, list[int]] = {}
        for aptes in self.param['aptes']:
            df_apt = self.residues_atoms[aptes]
            df_i = df_apt[(df_apt['atom_name'] == 'N') &
                          (df_apt['z'].between(zrange[0], zrange[1]))]
            # Get the 'residue_number' values for the filtered atoms
            aptes_index_dict[aptes] = df_i['residue_number'].values
        return aptes_index_dict

    def find_interface_z_range(self,
                               water_surface: 'WrapperGetSurface'
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

        z_range = self.calculate_interface_z_range(interface_z,
                                                   interface_w,
                                                   aptes_com)
        return z_range

    def calculate_interface_z_range(self,
                                    interface_z: float,  # Location of interfac
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
        elif self.param['LINE'] == 'DOUBLELOWERBOUND':
            self.info_msg += \
                '\tChecks APTES under the interface - standard deviation/2\n'
            z_range = (0, interface_z - interface_w + aptes_com)
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
        residues: list[str] = self.get_unique_residue_names()

        # Get the atoms for each residue and store them in a dictionary
        residues_atoms: dict[str, pd.DataFrame] = \
            self.__get_residues_atoms(residues)

        # Get the atoms in the bounding box enclosing the NP and add
        # them to the dictionary
        for aptes in self.param['aptes']:
            residues_atoms[f'box_{aptes}'] = \
                self.__get_np_box(residues_atoms, aptes)

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
                     residues_atoms: dict[str, pd.DataFrame],
                     aptes: str  # Name of the aptes chains
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
        # Get the x, y, and z ranges of the NP using  calculate_np_xyz_range
        xrange, yrange, zrange = \
            self.calculate_np_xyz_range(residues_atoms[aptes])
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
    def calculate_np_xyz_range(aptes_atoms: pd.DataFrame  # All APTES atoms
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

    def calculate_maximum_np_radius(self) -> np.float64:
        """get the maximum radius of NP, since APTES are most outward,
        here only looking at APTES residues"""
        np_diameters: list[np.float64] = []
        for aptes in self.param['aptes']:
            aptes_atoms: pd.DataFrame = self.residues_atoms[aptes]
            diameter: list[float] = []  # Save the diameters in each direction
            for xyz in ['x', 'y', 'z']:
                diameter.append(
                    aptes_atoms[xyz].max() - aptes_atoms[xyz].min())
            np_diameters.append(np.max(diameter))
        max_diameter: np.float64 = np.max(np_diameters)
        self.info_msg += \
            f'\tMaximum radius of between all NPs: `{max_diameter/2}`\n'
        return max_diameter

    def get_unique_residue_names(self) -> list[str]:
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
        """
        Write and log messages.

        Parameters:
            log (Logger): The logger object to log the messages.
        """
        print(f'{bcolors.OKCYAN}{ProcessData.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    data = ProcessData(sys.argv[1], log=logger.setup_logger('get_data.log'))
