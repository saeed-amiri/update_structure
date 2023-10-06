"""
IonizationSol: A script to incorporate counterions into the system by
searching for unoccupied spaces within the water section and identifying
all present atoms. It ensures that the placement of counterions does
not overlap with any existing atoms based on the number of new
protonation.

This script utilizes the Protonating class to find the positions for
new ions within the water phase. The number of ions and their locations
are determined based on the number of deprotonations of the APTES
chains.

Classes:
    IonizationSol: A class representing the process of ionizing the
    water phase of the box.

Methods:
    __init__(self, fname: str) -> None:
        Initialize the IonizationSol object.

    mk_ionization(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        Find the numbers of ions and their locations in the water phase.

    __check_poses(self, ion_poses: list[np.ndarray]) -> np.ndarray:
        Check for probable overlapping of positions and drop positions
        that are too close to each other.

    __best_ion_selction(self, poses: np.ndarray, velos: np.ndarray,
                        d_ions: list[float], n_portons: int) ->
                        tuple[list[np.ndarray], list[np.ndarray]]:
        Select random positions with respect to velocities for the ions.

    __drop_near_ions(atoms: np.ndarray, distance: float, tree: KDTree)
                    -> np.ndarray:
        Drop the positions which are very close to each other.

    __get_chunk_atoms(self, sol_atoms: pd.DataFrame,
                      x_chunks: list[tuple[np.float64, np.float64]],
                      y_chunks: list[tuple[np.float64, np.float64]],
                      z_chunks: list[tuple[np.float64, np.float64]])
                      -> tuple[list[np.ndarray], list[np.ndarray],
                                list[float]]:
        Get atoms within each chunk box and return a list of all
        possible positions for ions.

    _process_chunk_box(self, sol_atoms: pd.DataFrame, x_i: np.ndarray,
                       y_i: np.ndarray, z_i: np.ndarray) ->
                       tuple[np.ndarray, np.ndarray, float]:
        Process the chunk box, getting atoms in each box, and find
        positions for all the needed ions.

    find_ion_velocity(velocities: np.ndarray) -> np.ndarray:
        Find the average velocity of the water molecules in the
        section and return their mean as the velocity of the ion.

    find_position_with_min_distance(self, atoms: np.ndarray,
                                    box_dims: tuple[np.ndarray,
                                    np.ndarray, np.ndarray]) ->
                                    tuple[np.ndarray, float]:
        Find the best place for ions in each box.

    __get_chunk_interval(self, x_dims: np.ndarray, y_dims: np.ndarray,
    z_dims: np.ndarray, n_protons: int) -> tuple[list[tuple[np.float64,
    np.float64]], list[tuple[np.float64, np.float64]],
    list[tuple[np.float64, np.float64]]]:
        Get dimensions of each chunk box and return the dimensions of
        each chunk box.

    __get_chunk_numbers(self, chunk_num: int, n_protons: int) ->
                        tuple[int, int, int]:
        Find the best numbers for dividing the box into chunks.

    __get_axis_chunk(dims: np.ndarray, lims: float, chunk_num: int) ->
    list[tuple[np.float64, np.float64]]:
        Return the chunks of the axis.

    __get_box_size(sol_atoms: pd.DataFrame) -> tuple[np.ndarray,
                                                     np.ndarray,
                                                     np.ndarray]:
        Get the dimension of the sol box (water and ions).

    __get_sol_phase_atoms(self) -> pd.DataFrame:
        Get all the atoms below the interface, in the water section.

    __write_msg(self, log: logger.logging.Logger):
        Write and log messages.
"""


import sys
import operator
import multiprocessing as multip
from scipy.spatial import cKDTree, KDTree
import numpy as np
import pandas as pd
import protonating as proton
import logger
from colors_text import TextColor as bcolors


class IonizationSol(proton.FindHPosition):
    """
    The IonizationSol class ionizes the water phase of the system by
    placing counterions in unoccupied spaces within the water section,
    ensuring no overlap with existing atoms based on the number of new
    protonations.
    """

    info_msg: str  # Message to pass for logging and writing
    ion_poses: list[np.ndarray]  # Positoin for ions
    ion_velos: list[np.ndarray]  # Velocity for ions

    def __init__(self,
                 fname: str,  # Name of the pdb file
                 log: logger.logging.Logger
                 ) -> None:
        super().__init__(fname, log)
        self.info_msg = 'Message:\n'
        self.info_msg += '\tFinding poistions for new ions\n'
        self.ion_poses, self.ion_velos = self.ionize_water()
        self.__write_msg(log)
        self.info_msg = ''  # clean the msg

    def ionize_water(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """get the numbers of ions and their locations in the water
        phase"""
        x_dims: np.ndarray  # Dimensions of the sol box in x
        y_dims: np.ndarray  # Dimensions of the sol box in y
        z_dims: np.ndarray  # Dimensions of the sol box in z
        x_chunks: list[tuple[np.float64, np.float64]]
        y_chunks: list[tuple[np.float64, np.float64]]
        z_chunks: list[tuple[np.float64, np.float64]]
        ion_poses_list: list[np.ndarray]  # Possible position for ions
        ion_velos_list: list[np.ndarray]  # Possible velocity for ions
        ion_poses: list[np.ndarray]  # Main poistions for the ions
        ion_velos: list[np.ndarray]  # Main velocities for the ions
        d_ions: list[float]  # Distance of the new ions with their nighbours

        # Find the all the atoms in the water (sol) phase
        sol_atoms: pd.DataFrame = self.__get_sol_phase_atoms()

        # Get the dimension of the ares
        x_dims, y_dims, z_dims = self.__get_box_size(sol_atoms)

        # Get the number of the chains to be protonated
        n_protonation: int  # Number of the protonation
        n_protonation = sum(len(lst) for lst in self.unprot_aptes_ind.values())
        # Get the chunk boxes to find atoms in them
        x_chunks, y_chunks, z_chunks = \
            self.__get_chunk_interval(x_dims, y_dims, z_dims, n_protonation)

        # Find possible poistions for all the ions
        ion_poses_list, ion_velos_list, d_ions = \
            self.__get_chunk_atoms(sol_atoms, x_chunks, y_chunks, z_chunks)

        # Sanity check of the ions_positions
        ion_poses_tmp: np.ndarray = self.__check_poses(ion_poses_list)
        # Get the ions poistion based on the protonation
        ion_poses, ion_velos = \
            self.__best_ion_selction(ion_poses_tmp,
                                     np.array(ion_velos_list),
                                     d_ions,
                                     n_protonation)
        return ion_poses, ion_velos

    def __check_poses(self,
                      ion_poses: list[np.ndarray]  # Possible position for ions
                      ) -> np.ndarray:
        """
        Check for probable overlapping positions.

        Parameters:
            ion_poses (list[np.ndarray]): List of possible positions
            for ions.

        Returns:
            np.ndarray: Array of non-overlapping ion positions.
        """
        atoms: np.ndarray = np.vstack(ion_poses)

        # Build a KDTree from the atom coordinates
        tree = KDTree(atoms)

        # Query the tree for the closest pair of atoms
        min_distance, _ = tree.query(atoms, k=2, workers=12)

        # Return the indices and the minimum distance
        if np.min(min_distance[:, 1]) < self.param['ION_DISTANCE']:
            atoms = \
                self.__drop_near_ions(atoms, self.param['ION_DISTANCE'], tree)
        return atoms

    def __best_ion_selction(self,
                            poses: np.ndarray,  # All the positions for ions
                            velos: np.ndarray,  # All the velocities for ions
                            d_ions: list[float],  # Distnce of ions
                            n_portons: int  # Number of protonations
                            ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Select random positions with respect to velocities for the ions.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]:
                A tuple containing two lists:
                - List of selected positions for ions (list[np.ndarray]).
                - List of selected velocities for ions (list[np.ndarray]).
        """
        self.info_msg += f'\tTotal number of new ions is: `{n_portons}`\n'
        if len(poses) < n_portons:
            sys.exit(f'{bcolors.FAIL}{self.__module__}:\n'
                     f'\t Number of ions positoins: `{len(poses)}` is smaller'
                     f' than protonation: `{n_portons}`{bcolors.ENDC}\n')
        else:
            # Combine the two lists using zip
            combined = list(zip(d_ions, poses, velos))

            # Sort the combined list based on the values in descending order
            combined_sorted = sorted(combined,
                                     key=operator.itemgetter(0),
                                     reverse=True)

            # Select the m maximum values and their corresponding arrays
            selected_d = [item[0] for item in combined_sorted[:n_portons]]
            selected_poses = [item[1] for item in combined_sorted[:n_portons]]
            selected_velos = [item[2] for item in combined_sorted[:n_portons]]
            self.info_msg += \
                '\n\tThe selected spots have nighbours distance:\n'
            self.info_msg += \
                ", ".join(["{:.2f}".format(num) for num in selected_d])
        return selected_poses, selected_velos

    @staticmethod
    def __drop_near_ions(atoms: np.ndarray,  # All the ion poistions
                         distance: float,  # Minimum distance between ions
                         tree: KDTree  # From the atom coordinates
                         ) -> np.ndarray:
        """drop the poistions which are very colse to eachother"""

        # Query the tree for atoms within distance d
        atom_indices = tree.query_ball_tree(tree, r=distance)

        # Remove duplicate indices
        unique_indices = np.unique(np.concatenate(atom_indices))
        return atoms[unique_indices]

    def __get_chunk_atoms(self,
                          sol_atoms: pd.DataFrame,  # All Sol phase atoms
                          x_chunks: list[tuple[np.float64, np.float64]],  # rng
                          y_chunks: list[tuple[np.float64, np.float64]],  # rng
                          z_chunks: list[tuple[np.float64, np.float64]],  # rng
                          ) -> tuple[list[np.ndarray],
                                     list[np.ndarray],
                                     list[float]]:
        """get atoms within each chunk box, and return a list of all
        possible positions for ions. it could be more than the number
        of number of the new protonation."""
        chunks: list[tuple[pd.DataFrame,
                           tuple[np.float64, np.float64],
                           tuple[np.float64, np.float64],
                           tuple[np.float64, np.float64]]]
        chunks = [(sol_atoms, x_i, y_i, z_i)
                  for x_i in x_chunks
                  for y_i in y_chunks
                  for z_i in z_chunks]

        if len(chunks) > self.core_nr:
            num_processes = self.core_nr
        else:
            num_processes = len(chunks)
        with multip.Pool(processes=num_processes) as pool:
            ion_info: list[tuple[np.ndarray, np.ndarray, float]] = \
                pool.starmap(self._process_chunk_box, chunks)
        ion_poses: list[np.ndarray] = [item[0] for item in ion_info]
        ion_velos: list[np.ndarray] = [item[1] for item in ion_info]
        d_ions: list[float] = [item[2] for item in ion_info]
        return ion_poses, ion_velos, d_ions

    def _process_chunk_box(self,
                           sol_atoms: pd.DataFrame,  # All Sol phase atoms
                           x_i: np.ndarray,  # interval for the box
                           y_i: np.ndarray,  # interval for the box
                           z_i: np.ndarray  # interval for the box
                           ) -> tuple[np.ndarray, np.ndarray, float]:
        """process the chunk box, getting atoms in each box, and find
        positions for all the needed ions"""
        df_i = sol_atoms[(sol_atoms['x'] >= x_i[0]) &
                         (sol_atoms['x'] < x_i[1]) &
                         (sol_atoms['y'] >= y_i[0]) &
                         (sol_atoms['y'] < y_i[1]) &
                         (sol_atoms['z'] >= z_i[0]) &
                         (sol_atoms['z'] < z_i[1])
                         ]
        coordinates: np.ndarray = df_i[['x', 'y', 'z']].values
        try:
            velocities: np.ndarray = df_i[['vx', 'vy', 'vz']].values
        except ValueError:
            velocities = np.zeros(coordinates.shape)
        ion_vel: np.ndarray = self.find_ion_velocity(velocities)
        box_dims = (x_i, y_i, z_i)
        ion_pos: np.ndarray  # Ion positions
        d_ion: float  # Minimum distance between ion the box
        ion_pos, d_ion = \
            self.find_position_with_min_distance(coordinates,
                                                 box_dims)
        return (ion_pos, ion_vel, d_ion)

    @staticmethod
    def find_ion_velocity(velocities: np.ndarray  # Velocities of atom in df
                          ) -> np.ndarray:
        """find average velocity of the df section and return their
        mean as a velocity of the ion"""
        return np.mean(velocities, axis=0)

    def find_position_with_min_distance(self,
                                        atoms: np.ndarray,  # Coords of atoms
                                        box_dims: tuple[np.ndarray,
                                                        np.ndarray,
                                                        np.ndarray],  # Boxdims
                                        ) -> tuple[np.ndarray, float]:
        """
        Find the best place for ions in each box.

        Parameters:
            atoms (np.ndarray): Array of the coordinates of atoms.
            box_dims (tuple[np.ndarray, np.ndarray, np.ndarray]):
            Dimensions of the box.

        Returns:
            tuple[np.ndarray, float]:
                A tuple containing two values:
                - Array of ion positions (np.ndarray).
                - Minimum distance between ion and the box (float).
        """

        min_x, max_x = box_dims[0]
        min_y, max_y = box_dims[1]
        min_z, max_z = box_dims[2]

        # Build a kd-tree from the atom coordinates
        tree = cKDTree(atoms)

        positions: list[np.ndarray] = []  # Keep postions
        d_ions: list[float] = []  # Keep track of d_ions

        while len(positions) < int(self.param['BETTER_POS']):
            d_ion = self.param['ION_DISTANCE']  # Distance of Ions and others
            found_poistion: bool = False
            while d_ion > 0 and not found_poistion:
                for _ in range(int(self.param['ION_ATTEMPTS'])):
                    # Generate random positions within the specified ranges
                    # for each dimension, Don't want to be at the edge of box
                    position = \
                        np.random.uniform(low=[min_x, min_y, min_z],
                                          high=[max_x, max_y, max_z-1])

                    # Query the kd-tree for the nearest neighbors within d
                    distances, _ = tree.query(position,
                                              k=1,
                                              distance_upper_bound=d_ion)

                    # If any distance is less than d, continue to the next try
                    if np.all(distances >= d_ion):
                        positions.append(position)
                        d_ions.append(d_ion)
                        found_poistion = True
                        break
                d_ion -= 0.01

        if positions:
            return positions[np.argmax(d_ions)], d_ions[np.argmax(d_ions)]
        raise ValueError("Unable to find a suitable position within \
                          the specified constraints")

    def __get_chunk_interval(self,
                             x_dims: np.ndarray,  # Dimensions of sol box in x
                             y_dims: np.ndarray,  # Dimensions of sol box in y
                             z_dims: np.ndarray,  # Dimensions of sol box in z
                             n_protons: int  # Number of protonated chains
                             ) -> tuple[list[tuple[np.float64, np.float64]],
                                        list[tuple[np.float64, np.float64]],
                                        list[tuple[np.float64, np.float64]]]:
        """
        Get dimensions of each chunk box and return the dimensions of
        each chunk box.

        Parameters:
            x_dims (np.ndarray): Dimensions of the sol box in the x-axis.
            y_dims (np.ndarray): Dimensions of the sol box in the y-axis.
            z_dims (np.ndarray): Dimensions of the sol box in the z-axis.
            n_protons (int): Number of protonated chains.

        Returns:
                A tuple containing three lists:
                - List of chunk intervals along the x-axis
                - List of chunk intervals along the y-axis
                - List of chunk intervals along the z-axis
        """
        x_lim: float = x_dims[1] - x_dims[0]
        y_lim: float = y_dims[1] - y_dims[0]
        z_lim: float = z_dims[1] - z_dims[0]
        chunk_num: int = int(np.cbrt(n_protons))
        chunk_axis: tuple[int, int, int] = \
            self.__get_chunk_numbers(chunk_num, n_protons)
        x_chunks: list[tuple[np.float64, np.float64]] = \
            self.__get_axis_chunk(x_dims, x_lim, chunk_axis[0])  # Chunks range
        y_chunks: list[tuple[np.float64, np.float64]] = \
            self.__get_axis_chunk(y_dims, y_lim, chunk_axis[1])  # Chunks range
        z_chunks: list[tuple[np.float64, np.float64]] = \
            self.__get_axis_chunk(z_dims, z_lim, chunk_axis[2])  # Chunks range
        return x_chunks, y_chunks, z_chunks

    def __get_chunk_numbers(self,
                            chunk_num: int,  # initial chunk number in each ax
                            n_protons: int  # number of protonated chains
                            ) -> tuple[int, int, int]:
        """find best numbers for dividing the box into chunks"""
        chunck_axis: tuple[int, int, int]
        if chunk_num**3 == n_protons:
            chunck_axis = (chunk_num, chunk_num, chunk_num)
        elif chunk_num**3 < n_protons:
            x_chunk, y_chunk, z_chunk = chunk_num, chunk_num, chunk_num
            while x_chunk * y_chunk * z_chunk < n_protons:
                x_chunk += 1
            chunck_axis = x_chunk, y_chunk, z_chunk
        return chunck_axis

    @staticmethod
    def __get_axis_chunk(dims: np.ndarray,  # Dimensions of the axis
                         lims: float,  # Limites of the box
                         chunk_num: int  # Number of chunks
                         ) -> list[tuple[np.float64, np.float64]]:
        """return the chunks of the aixs"""
        chunk_intervals: list[tuple[np.float64, np.float64]] = []
        for i in range(chunk_num):
            x_0 = dims[0]+i*lims/chunk_num
            x_1 = x_0 + lims/chunk_num
            chunk_intervals.append((x_0, x_1))
        return chunk_intervals

    @staticmethod
    def __get_box_size(sol_atoms: pd.DataFrame  # All the atoms below NP
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """get the dimension of the sol box (water and ions)"""
        x_dims = np.array([np.min(sol_atoms['x']), np.max(sol_atoms['x'])])
        y_dims = np.array([np.min(sol_atoms['y']), np.max(sol_atoms['y'])])
        z_dims = np.array([np.min(sol_atoms['z']), np.max(sol_atoms['z'])])
        return x_dims, y_dims, z_dims

    def __get_sol_phase_atoms(self) -> pd.DataFrame:
        """get all the atom below interface, in the water section"""
        tresh_hold = float(self.param['INTERFACE']-self.np_diameter)
        return self.atoms[self.atoms['z'] < tresh_hold]

    def __write_msg(self,
                    log: logger.logging.Logger,  # To log info in it
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{IonizationSol.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    ioning = IonizationSol(sys.argv[1], log=logger.setup_logger('update.log'))
