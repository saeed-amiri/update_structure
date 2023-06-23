"""In order to incorporate counterions into the system, the script
searches for unoccupied spaces within the water section and identifies
all present atoms. It ensures that the placement of counterions does
not overlap with any existing atoms based on the number of new
protonation."""


import sys
import multiprocessing as multip
from scipy.spatial import cKDTree, KDTree
import numpy as np
import pandas as pd
import protonating as proton
from colors_text import TextColor as bcolors


class IonizationSol(proton.FindHPosition):
    """ionizing the water phase of the box. The number is equal to the
    number of deprotonation of the APTES"""

    ion_poses: np.ndarray  # Positoin for ions

    def __init__(self,
                 fname: str  # Name of the pdb file
                 ) -> None:
        super().__init__(fname)
        self.ion_poses = self.mk_ionization()

    def mk_ionization(self) -> np.ndarray:
        """get the numbers of ions and their locations in the water
        phase"""
        x_dims: np.ndarray  # Dimensions of the sol box in x
        y_dims: np.ndarray  # Dimensions of the sol box in y
        z_dims: np.ndarray  # Dimensions of the sol box in z
        x_chunks: list[tuple[np.float64, np.float64]]
        y_chunks: list[tuple[np.float64, np.float64]]
        z_chunks: list[tuple[np.float64, np.float64]]
        ion_poses_list: list[np.ndarray]  # Possible position for ions
        ion_poses: np.ndarray  # Main poistions for the ions
        # Find the all the atoms in the water (sol) phase
        sol_atoms: pd.DataFrame = self.__get_sol_phase_atoms()
        # Get the dimension of the ares
        x_dims, y_dims, z_dims = self.__get_box_size(sol_atoms)
        # Get the chunk boxes to find atoms in them
        x_chunks, y_chunks, z_chunks = \
            self.__get_chunk_interval(x_dims, y_dims, z_dims)
        # Find possible poistions for all the ions
        ion_poses_list = \
            self.__get_chunk_atoms(sol_atoms, x_chunks, y_chunks, z_chunks)
        # Sanity check of the ions_positions
        ion_poses = self.__check_poses(ion_poses_list)
        # Get the ions poistion based on the protonation
        ion_poses = \
            self.__random_pos_selction(ion_poses, len(self.unprot_aptes_ind))
        return ion_poses

    def __check_poses(self,
                      ion_poses: list[np.ndarray]  # Possible position for ions
                      ) -> np.ndarray:
        """check for probable ovrlapping of positions"""
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

    def __random_pos_selction(self,
                              atoms: np.ndarray,  # All the positions for ions
                              n_portons: int  # Number of protonations
                              ) -> np.ndarray:
        """select random positions for the ions"""
        if len(atoms) < n_portons:
            sys.exit(f'{bcolors.FAIL}{self.__module__}:\n'
                     f'\t Number of ions positoins: `{len(atoms)}` is smaller'
                     f' than protonation: `{n_portons}`')
        else:
            np.random.shuffle(atoms)  # Shuffle the rows randomly
            atoms = atoms[:n_portons]  # Select the first n rows
        return atoms

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
                          ) -> list[np.ndarray]:
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

        with multip.Pool() as pool:
            ion_poses: list[np.ndarray] = \
                pool.starmap(self._process_chunk_box, chunks)
        return ion_poses

    def _process_chunk_box(self,
                           sol_atoms: pd.DataFrame,  # All Sol phase atoms
                           x_i: np.ndarray,  # interval for the box
                           y_i: np.ndarray,  # interval for the box
                           z_i: np.ndarray  # interval for the box
                           ) -> pd.DataFrame:
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
        box_dims = (x_i, y_i, z_i)
        ion_pos: np.ndarray = \
            self.find_position_with_min_distance(coordinates,
                                                 box_dims)
        return ion_pos

    def find_position_with_min_distance(self,
                                        atoms: np.ndarray,  # Coords of atoms
                                        box_dims: tuple[np.ndarray,
                                                        np.ndarray,
                                                        np.ndarray],  # Boxdims
                                        ) -> np.ndarray:
        """find the best place for ions in each box"""
        d_ion = self.param['ION_DISTANCE']  # Distance of Ions and others
        num_attempts = int(self.param['ION_ATTEPTS'])  # Number to try to find

        min_x, max_x = box_dims[0]
        min_y, max_y = box_dims[1]
        min_z, max_z = box_dims[2]

        # Build a kd-tree from the atom coordinates
        tree = cKDTree(atoms)

        for _ in range(num_attempts):
            # Generate random positions within the specified ranges
            # for each dimension, Don't want to be at the edge of the box
            position = \
                np.random.uniform(low=[min_x, min_y, min_z],
                                  high=[max_x, max_y, max_z-1])

            # Query the kd-tree for the nearest neighbors within distance d
            _, distances = tree.query(position,
                                      k=1,
                                      distance_upper_bound=d_ion)

            # If any distance is less than d, continue to the next attempt
            if np.any(distances < d_ion):
                continue

            # If all distances are greater than or equal to d,return
            # the position
            return position

        # If no suitable position is found after all attempts, return None
        return np.array([-1, -1, -1])

    def __get_chunk_interval(self,
                             x_dims: np.ndarray,  # Dimensions of sol box in x
                             y_dims: np.ndarray,  # Dimensions of sol box in y
                             z_dims: np.ndarray  # Dimensions of sol box in z
                             ) -> tuple[list[tuple[np.float64, np.float64]],
                                        list[tuple[np.float64, np.float64]],
                                        list[tuple[np.float64, np.float64]]]:
        """get dimension of each chunk-box and return the dimension of
        each chunk-box"""
        x_lim: float = x_dims[1] - x_dims[0]
        y_lim: float = y_dims[1] - y_dims[0]
        z_lim: float = z_dims[1] - z_dims[0]
        chunk_num: int = int(np.cbrt(len(self.unprot_aptes_ind)))
        chunk_axis: tuple[int, int, int] = self.__get_chunk_numbers(chunk_num)
        x_chunks: list[tuple[np.float64, np.float64]] = \
            self.__get_axis_chunk(x_dims, x_lim, chunk_axis[0])  # Chunks range
        y_chunks: list[tuple[np.float64, np.float64]] = \
            self.__get_axis_chunk(y_dims, y_lim, chunk_axis[1])  # Chunks range
        z_chunks: list[tuple[np.float64, np.float64]] = \
            self.__get_axis_chunk(z_dims, z_lim, chunk_axis[2])  # Chunks range
        return x_chunks, y_chunks, z_chunks

    def __get_chunk_numbers(self,
                            chunk_num: int,  # initial chunk number in each ax
                            ) -> tuple[int, int, int]:
        """find best numbers for dividing the box into chunks"""
        proton_num: int = int(len(self.unprot_aptes_ind))
        chunck_axis: tuple[int, int, int]
        if chunk_num**3 == proton_num:
            chunck_axis = (chunk_num, chunk_num, chunk_num)
        elif chunk_num**3 < proton_num:
            x_chunk, y_chunk, z_chunk = chunk_num, chunk_num, chunk_num
            while x_chunk * y_chunk * z_chunk < proton_num:
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
        tresh_hold = float(self.param['INTERFACE']+100-self.np_diameter)
        return self.atoms[self.atoms['z'] < tresh_hold]


if __name__ == '__main__':
    ioning = IonizationSol(sys.argv[1])
