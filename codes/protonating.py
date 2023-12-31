"""This script utilizes get_data.py to obtain a list and data for the
APTES chain that requires protonation. The script then identifies
suitable positions for each additional hydrogen atom near the N atom,
ensuring it does not interfere with other atoms in the surrounding
residues. Initially, the script locates a plausible area for the
hydrogen atom and subsequently checks for the presence of other atoms.
Finally, the script places the hydrogen atom in a suitable position.
"""


import sys
import multiprocessing as multip
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import get_data
import logger
from colors_text import TextColor as bcolors


class FindHPosition(get_data.ProcessData):
    """Find an area in which the new H could set

    Attributes of get_data.ProcessData:

        atoms: pd.DataFrame  -> All atoms dataframe
        param: dict[str, Any]  -> All the parameters from input file
        residues_atoms: dict[str, pd.DataFrame]  # Atoms info for each
                                                   residue
        unproton_aptes: dict[str, pd.DataFrame]  # APTES to be protonated
        unprot_aptes_ind: dict[str, list[int]]  # Index of APTES to be
                                                  protonated
        np_diameter: np.float64  -> Diameter of NP, based on APTES
                                    positions
        title: str  -> Name of the system; if the file is gro
        pbc_box: str  -> PBC of the system; if the file is gro
    """

    info_msg: str  # Message to pass for logging and writing
    h_porotonations: dict[str, dict[int, np.ndarray]] = {}
    h_velocities: dict[str, dict[int, np.ndarray]] = {}

    def __init__(self,
                 fname: str,  # Name of the pdb file
                 log: logger.logging.Logger
                 ) -> None:
        super().__init__(fname, log)
        self.h_porotonations, self.h_velocities = self.get_area()
        self.info_msg = 'Message:\n'
        self.info_msg += '\tFinding poistions for new HN3 atoms\n'
        self.__write_msg(log)
        self.info_msg = ''  # clean the msg

    def get_area(self) -> tuple[dict[str, dict[int, np.ndarray]],
                                dict[str, dict[int, np.ndarray]]]:
        """
        Find an area around the N, forming a cone with an angle equal
        to the angle HN1-N-HN2, and compute the velocity for the new
        H atom.

        Returns:
            tuple[dict[str, dict[int, np.ndarray]],
                  dict[str, dict[int, np.ndarray]]]:
            A tuple containing two dictionaries:
            1. 'h_porotonations': A nested dictionary mapping APTES
            names to residue indices and the corresponding new H atom
            positions.
            2. 'h_velocities': A nested dictionary mapping APTES names
            to residue indices and the corresponding velocities for the
            new H atom.

        Note:
            - The method processes the APTES residues in parallel
            using multiple processes to speed up calculations.
            - For each APTES, the method iterates through its associated
            residue indices and calculates the positions and velocities
            of new H atoms based on the N and HN1/HN2 atoms' positions.
            - The 'h_porotonations' dictionary maps APTES names to
            dictionaries, where each inner dictionary maps residue
            indices to the corresponding positions of new H atoms.
            - The 'h_velocities' dictionary maps APTES names to
            dictionaries, where each inner dictionary maps residue
            indices to the corresponding velocities of new H atoms.
        """
        # All the H locations wtih index
        results: list[tuple[dict[int, np.ndarray], dict[int, np.ndarray]]]
        
        h_porotonations: dict[str, dict[int, np.ndarray]] = {}  # dicts of locs
        h_velocities: dict[str, dict[int, np.ndarray]] = {}  # dicts of velocs
        for aptes, items in self.unprot_aptes_ind.items():
            if len(items) > self.core_nr:
                num_processes = self.core_nr
            else:
                num_processes = len(items)
            chunk_size: int = len(items) // num_processes
            chunks = [items[i:i+chunk_size] for i in
                      range(0, len(items), chunk_size)]
            with multip.Pool(processes=num_processes) as pool:
                results = pool.starmap(
                    self.process_ind, [(chunk, aptes) for chunk in chunks])
            h_porotonations_i: dict[int, np.ndarray] = {}
            h_velocities_i: dict[int, np.ndarray] = {}
            for item in results:
                h_porotonations_i.update(item[0])
                h_velocities_i.update(item[1])
            h_porotonations[aptes] = h_porotonations_i
            h_velocities[aptes] = h_velocities_i
        return h_porotonations, h_velocities

    def process_ind(self,
                    chunk: list[int],  # Chunk of the indices for loop over
                    aptes: str  # Name of the APTES
                    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """
        Perform the necessary calculations for a chunk of residue
        indices in an APTES.

        Parameters:
            chunk (list[int]): A list of residue indices for which the
            calculations will be performed.
            aptes (str): The name of the APTES.

        Returns:
            tuple[dict[int, np.ndarray], dict[int, np.ndarray]]: A
            tuple containing two dictionaries:
            1. A dictionary mapping residue indices to the corresponding
            positions of new H atoms ('all_h_locs').
            2. A dictionary mapping residue indices to the corresponding
            velocities of new H atoms ('all_h_vels').

        Note:
            - The method calculates the positions and velocities of new
            H atoms for the provided chunk of residue indices in a
            specific APTES.
            - The 'all_h_locs' dictionary maps residue indices to the
            positions of new H atoms in the coordinate space.
            - The 'all_h_vels' dictionary maps residue indices to the
            velocities of new H atoms.
        """

        df_i: pd.DataFrame  # Local df for each residue index
        df_nh: pd.DataFrame  # Local df for each residue index
        v_nh1: np.ndarray  # Vector from N to H1
        v_nh2: np.ndarray  # Vector from N to H2
        v_mean: np.float64  # Mean length of the NH bonds
        # nh_angle: np.float64  # Angle between the two NH bonds, in radians
        atoms_around_n: pd.DataFrame  # Atoms in a radius of N
        possible_loc: list[np.ndarray]  # Possible locations for putting H
        h_loc: np.ndarray  # Possible location for H
        all_h_locs: dict[int, np.ndarray] = {}  # To save all the locations
        all_h_vels: dict[int, np.ndarray] = {}  # To save all the velocities

        for ind in chunk:
            df_i = self.unproton_aptes[aptes][
                self.unproton_aptes[aptes]['residue_number'] == ind]
            df_nh = df_i[df_i['atom_name'].isin(['N', 'HN1', 'HN2'])]
            v_nh1, v_nh2 = self.__get_vectors(df_nh)
            if 'vx' in df_nh.columns:
                all_h_vels[ind] = self.__get_velocity(df_nh)
            v_mean, _ = self.__get_hbond_len_angle(v_nh1, v_nh2)
            atoms_around_n = self.__get_atoms_around_n(df_nh, v_mean, aptes)
            possible_loc = self.__get_possible_pos(v_nh1, v_nh2, v_mean, df_nh)
            h_loc = self.__find_h_place(atoms_around_n, possible_loc, v_mean)
            all_h_locs[ind] = h_loc
        return (all_h_locs, all_h_vels)

    def __get_velocity(self,
                       df_nh: pd.DataFrame  # Contains only N and NHs
                       ) -> np.ndarray:
        """get the velocity for the new HN3"""
        vec: dict[str, np.ndarray] = {}
        m_h: float = 1.008  # Mass of atoms
        m_n: float = 14.007  # Mass of atoms
        total_mass: float = 2 * m_h + m_n

        # Calculate the center of mass velocity of the initial three atoms
        vec = self.__get_velocityxyz(df_nh)
        com_velocity = [(m_h * vec['HN1'][i] +
                         m_h * vec['HN2'][i] +
                         m_n * vec['N'][i]) / total_mass for i in range(3)]
        return np.array(com_velocity)

    @staticmethod
    def __get_velocityxyz(df_nh: pd.DataFrame  # Contains only N and NHs
                          ) -> dict[str, np.ndarray]:
        """return velocity (xyz) of the atoms"""
        vec_dict: dict[str, np.ndarray] = {}
        for _, row in df_nh.iterrows():
            vec = np.array([float(row['vx']),
                            float(row['vy']),
                            float(row['vz'])])
            vec_dict[row['atom_name']] = vec
        return vec_dict

    def __find_h_place(self,
                       atoms_around_n: pd.DataFrame,  # Atoms in radius of N
                       possible_loc: list[np.ndarray],  # Possible for H
                       v_mean: np.float64  # Mean of N-H bonds
                       ) -> np.ndarray:
        """
        Attempt to find the best position for the H atom among the
        possible positions. This method iterates through the list of
        possible positions for the H atom and checks if each position
        meets the condition of being within the specified radius 'v_mean'
        from the N atom. It verifies whether the position avoids overlaps
        with existing atoms in the 'atoms_around_n' DataFrame.

        Parameters:
            atoms_around_n (pd.DataFrame): DataFrame containing atoms
            within a specified radius of the N atom.
            possible_loc (List[np.ndarray]): A list of NumPy arrays
            representing the possible positions for the H atom around
            the N atom.
            v_mean (np.float64): The mean length of N-H bonds.

        Returns:
            np.ndarray: A NumPy array representing the best position
            found for the H atom around the N atom.

        Note:
            - The method iterates through the 'possible_loc' list to
            find the best position for the H atom.
            - It checks if each position in 'possible_loc' meets the
            condition of being at least 'v_mean' away from any atom in
            the 'atoms_around_n' DataFrame. If a position satisfies the
            condition, it is considered as a potential position for the
            H atom.
            - The 'in_flag' variable is used to track whether a position
            meets the condition or not. If a position does not meet the
            condition, the method breaks out of the inner loop and checks
            the next position.
            - If a position is found that satisfies the condition and
            avoids overlaps with existing atoms, it is returned as the
            best position for the H atom.
            - If no suitable position is found, the method raises a
            'SystemExit' exception with an error message indicating that a
            location for the H atom could not be found.

        Example:
            # DataFrame containing atoms within the radius of the N atom
            atoms_around_n = pd.DataFrame({
                'atom_name': ['N', 'H', 'C', 'H', 'H', ...],
                'x': [0.0, 1.0, 0.5, 1.5, 0.5, ...],
                'y': [0.0, 0.0, 0.5, 0.5, 1.5, ...],
                'z': [0.0, 0.0, 1.0, 1.5, 0.5, ...],
                ...
            })

            # List of possible positions for the H atom
            possible_loc = \
                [np.array([0.5, 0.5, 0.5]), np.array([1.0, 1.0, 1.0]), ...]

            # Mean length of N-H bonds
            v_mean = 1.0

            # Find the best position for the H atom
            best_position = \
                ProcessData.__find_h_place(
                    atoms_around_n, possible_loc, v_mean)
        """

        in_flag: bool  # To check if the point meets the condition
        loc: np.ndarray = np.array([-1, -1, -1])  # initial point, CAN'T BE -1
        for loc in possible_loc:
            in_flag = True
            for _, row in atoms_around_n.iterrows():
                atom_i = np.array([row['x'], row['y'], row['z']])
                distance = np.linalg.norm(loc-atom_i)
                if distance >= v_mean:
                    break
                in_flag = False
            if not in_flag:
                break
        if not in_flag:
            sys.exit(f'{bcolors.FAIL}Called from {self.__module__}:'
                     ' (protonation.py)\n'
                     '\tError! Could not find a location for H atom\n'
                     f'{bcolors.ENDC}')
        return loc

    def __get_possible_pos(self,
                           v_nh1: np.ndarray,  # Vector from N to H1
                           v_nh2: np.ndarray,  # Vector from N to H2
                           v_mean: np.float64,  # Length of N-H bond
                           df_nh: pd.DataFrame,  # N and H infos
                           ) -> list[np.ndarray]:
        """
        Find all the points that satisfy the given angle and distance
        conditions. This method generates a list of possible positions
        around the N atom for new H atoms based on the vectors from N
        to H1 and N to H2. The vectors are rotated around the axis
        defined by the cross product of the normalized vectors.
        The generated points are then normalized and scaled to the
        average length of N-H bonds. The method considers a specified
        number of samples to ensure uniform coverage around the N atom.

        Parameters:
            v_nh1 (np.ndarray): Vector from N to H1.
            v_nh2 (np.ndarray): Vector from N to H2.
            v_mean (np.float64): The average length of N-H bonds.
            df_nh (pd.DataFrame): DataFrame containing N and H atoms
            data.

        Returns:
            List[np.ndarray]: A list of NumPy arrays representing the
            possible positions for new H atoms around the N atom.

        Note:
            - The method efficiently uses NumPy functions to normalize
            the input vectors 'v_nh1' and 'v_nh2'.
            - The axis of rotation is computed as the cross product of
            the normalized vectors 'v_nh1' and 'v_nh2'.
            - The 'num_samples' parameter, defined as
            'self.param['NUMSAMPLE']',
            represents the number of points to generate around the N
            atom.
            - The method extracts the position of the N atom from the
            DataFrame 'df_nh' using the 'atom_name' column and constructs
            a list 'n_pos' containing its x, y, and z coordinates.
            - The method generates 'num_samples' vectors by incrementally
            rotating the vector 'v_nh1' around the computed axis.
            Each vector is normalized and scaled to the average length
            of N-H bonds, and its position is adjusted based on the
            position of the N atom.
            - The method returns a list of NumPy arrays representing
            the possible positions for new H atoms around the N atom.

        Example:
            # Vector from N to H1
            v_nh1 = np.array([1.0, 0.0, 0.0])

            # Vector from N to H2
            v_nh2 = np.array([0.0, 1.0, 0.0])

            # Average length of N-H bonds
            v_mean = 1.0

            # DataFrame containing N and H atoms
            df_nh = pd.DataFrame({
                'atom_name': ['N', 'H', 'C', 'H', 'H', ...],
                'x': [0.0, 1.0, 0.5, 1.5, 0.5, ...],
                'y': [0.0, 0.0, 0.5, 0.5, 1.5, ...],
                'z': [0.0, 0.0, 1.0, 1.5, 0.5, ...],
                ...
            })

            # Generate a list of possible positions for new H atoms
            possible_positions = \
                ProcessData.__get_possible_pos(v_nh1, v_nh2, v_mean, df_nh)
        """

        num_samples: int = int(self.param['NUMSAMPLE'])  # How many points
        # Normalize the vectors
        v1_norm: np.ndarray = v_nh1 / np.linalg.norm(v_nh1)
        v2_norm: np.ndarray = v_nh2 / np.linalg.norm(v_nh2)

        # Compute the axis of rotation
        axis: np.ndarray = np.cross(v1_norm, v2_norm)

        # Initialize the list of vectors
        vectors: list[np.ndarray] = []
        n_pos = [float(df_nh[df_nh['atom_name'] == 'N']['x'].iloc[0]),
                 float(df_nh[df_nh['atom_name'] == 'N']['y'].iloc[0]),
                 float(df_nh[df_nh['atom_name'] == 'N']['z'].iloc[0])]

        # Generate the vectors
        for i in range(num_samples):
            # increment: 2 * np.pi / num_samples
            angle = i * 2 * np.pi / num_samples
            rotated_vector = self.rotate_vector(v1_norm, axis, angle)
            normalized_vector = rotated_vector / np.linalg.norm(rotated_vector)
            vector_with_length_d = normalized_vector * v_mean
            vector_with_length_d += n_pos
            vectors.append(vector_with_length_d)
        return vectors

    @staticmethod
    def rotate_vector(vector: np.ndarray,  # Normelized vector of one N-H
                      axis: np.ndarray,  # Direction of the axis to rotate
                      angle: float  # Angle for rotation
                      ) -> list[np.ndarray]:
        """
        Rotate the given vector around the specified axis by the given
        angle.

        Parameters:
            vector (np.ndarray): The normalized vector representing the
            direction of rotation.
            axis (np.ndarray): The normalized vector representing the
            axis of rotation.
            angle (float): The angle in radians for rotation.

        Returns:
            list[np.ndarray]: A list containing all possible rotated
            vectors.

        Note:
            - The input 'vector' and 'axis' must be normalized before
            calling this
            method to ensure accurate results.
            - The method returns a list of all possible rotated vectors
            obtained by applying the rotation matrix to the input
            'vector'.
            - The rotation matrix is computed using the Rodrigues'
            rotation formula.
            - The method is intended for 3D vector rotation.
            - The 'cos_theta' and 'sin_theta' values are precomputed
            to optimize performance during the rotation calculation.
            - The method is static and can be called directly on the
            class.

        Example:
            # Define the normalized vector for rotation
            vector = np.array([1.0, 0.0, 0.0])

            # Define the normalized axis of rotation
            axis = np.array([0.0, 0.0, 1.0])

            # Define the angle for rotation (in radians)
            angle = np.pi / 2

            # Rotate the vector around the axis by the given angle
            rotated_vectors = \
                ProcessData.rotate_vector(vector, axis, angle)
        """

        # Normalize the axis vector
        axis = axis / np.linalg.norm(axis)

        # Compute the rotation matrix
        cos_theta: np.float64 = np.cos(angle)
        sin_theta: np.float64 = np.sin(angle)
        rot_matrix = \
            np.array([
                      [cos_theta + axis[0]**2*(1 - cos_theta),
                       axis[0]*axis[1]*(1 - cos_theta) - axis[2]*sin_theta,
                       axis[0]*axis[2]*(1 - cos_theta) + axis[1]*sin_theta],

                      [axis[1]*axis[0]*(1 - cos_theta) + axis[2]*sin_theta,
                       cos_theta + axis[1]**2*(1 - cos_theta),
                       axis[1]*axis[2]*(1 - cos_theta) - axis[0]*sin_theta],

                      [axis[2]*axis[0]*(1 - cos_theta) - axis[1]*sin_theta,
                       axis[2]*axis[1]*(1 - cos_theta) + axis[0]*sin_theta,
                       cos_theta + axis[2]**2*(1 - cos_theta)]
                     ])

        # Rotate the vector
        rotated_vector = np.dot(rot_matrix, vector)

        return rotated_vector

    def __get_atoms_around_n(self,
                             df_nh: pd.DataFrame,  # N and H dataframe
                             v_mean: np.float64,  # Average length of N-H bonds
                             aptes: str  # Name of the aptes chains
                             ) -> pd.DataFrame:
        """
        Generate a DataFrame that includes atoms within a specified
        radius around N atoms. This method efficiently checks for
        overlaps with new H atoms by only considering the relevant
        atoms. The analysis is limited to the atoms in the box enclosing
        the NP, which contains significantly fewer atoms than the entire
        system.

        Parameters:
            df_nh (pd.DataFrame): DataFrame containing N and H atoms
            data.
            v_mean (np.float64): The average length of N-H bonds.

        Returns:
            pd.DataFrame: DataFrame containing atoms within the
            specified radius around N atoms.

        Note:
            - The method efficiently uses a KD-tree data structure to
            find the atoms within the specified radius.
            - The input DataFrame 'df_nh' must contain data for N and H
            atoms to find the atoms around N.
            - The 'v_mean' parameter represents the average length of
            N-H bonds and is used to determine the search radius.
            - The method extracts the x, y, and z coordinates from the
            DataFrame 'self.residues_atoms['box']' that contains atoms
            in the bounding box enclosing the NP, ensuring a more
            efficient search.
            - The method calculates the search radius as 'v_mean + 2'
            to account for possible variations in bond lengths and
            provide a buffer around N atoms.
            - The KD-tree efficiently queries the points within the
            radius and returns the indices of the atoms within the
            radius.
            - The method returns the DataFrame containing the atoms
            within the specified radius around N atoms, which can be
            used for checking overlaps with new H atoms.

        Example:
            # DataFrame containing N and H atoms
            df_nh = pd.DataFrame({
                'atom_name': ['N', 'H', 'C', 'H', 'H', ...],
                'x': [0.0, 1.0, 0.5, 1.5, 0.5, ...],
                'y': [0.0, 0.0, 0.5, 0.5, 1.5, ...],
                'z': [0.0, 0.0, 1.0, 1.5, 0.5, ...],
                ...
            })

            # Define the average length of N-H bonds
            v_mean = 1.0

            # Generate the DataFrame with atoms around N within the
            # radius
            atoms_around_n = \
                ProcessData.__get_atoms_around_n(df_nh, v_mean)
        """
        # Extract the x, y, z coordinates from the dataframe
        box: str = f'box_{aptes}'
        coordinates: np.ndarray = \
            self.residues_atoms[box][['x', 'y', 'z']].values
        # Build the KD-tree
        tree = KDTree(coordinates)
        # Getting the position of the N atom
        n_pos = [float(df_nh[df_nh['atom_name'] == 'N']['x'].iloc[0]),
                 float(df_nh[df_nh['atom_name'] == 'N']['y'].iloc[0]),
                 float(df_nh[df_nh['atom_name'] == 'N']['z'].iloc[0])]

        radius: float = float(v_mean) + 2
        # Find the indices of points within the radius
        indices: list[int] = tree.query_ball_point(n_pos, radius)
        # Return the points within the radius
        return self.residues_atoms[box].iloc[indices]

    @staticmethod
    def __get_hbond_len_angle(v_nh1: np.ndarray,  # Vector from N to H1
                              v_nh2: np.ndarray  # Vector from N to H2
                              ) -> tuple[np.float64, np.float64]:
        """calculate the N-H bonds length and return their average
        also, get the angle between them"""
        v1_len: np.float64 = np.linalg.norm(v_nh1)
        v2_len: np.float64 = np.linalg.norm(v_nh2)
        dot_product: np.float64 = np.dot(v_nh1, v_nh2)
        angle_rad: np.float64 = np.arccos(dot_product / (v1_len * v2_len))
        return np.mean([v1_len, v2_len]), angle_rad

    def __get_vectors(self,
                      df_nh: pd.DataFrame  # Contains only N and NHs
                      ) -> tuple[np.ndarray, np.ndarray]:
        """find the vectors between N and H atoms"""
        vec: dict[str, np.ndarray] = {}
        vec = self.__get_posixyz(df_nh)
        #  Calculate vectors from N to NH1 and NH2
        v_nh1 = np.array([vec['HN1'][0] - vec['N'][0],
                          vec['HN1'][1] - vec['N'][1],
                          vec['HN1'][2] - vec['N'][2]])
        v_nh2 = np.array([vec['HN2'][0] - vec['N'][0],
                          vec['HN2'][1] - vec['N'][1],
                          vec['HN2'][2] - vec['N'][2]])
        return v_nh1, v_nh2

    @staticmethod
    def __get_posixyz(df_nh: pd.DataFrame  # Contains only N and NHs
                      ) -> dict[str, np.ndarray]:
        """return position (xyz) of the atoms"""
        vec_dict: dict[str, np.ndarray] = {}
        for _, row in df_nh.iterrows():
            vec = np.array([float(row['x']), float(row['y']), float(row['z'])])
            vec_dict[row['atom_name']] = vec
        return vec_dict

    def __write_msg(self,
                    log: logger.logging.Logger,  # To log info in it
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{FindHPosition.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    FindHPosition(sys.argv[1], log=logger.setup_logger('update.log'))
