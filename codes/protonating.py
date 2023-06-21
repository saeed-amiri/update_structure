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


class FindHPosition(get_data.ProcessData):
    """Find an area in which the new H could set"""
    def __init__(self,
                 fname: str  # Name of the pdb file
                 ) -> None:
        super().__init__(fname)
        self.get_area()

    def get_area(self) -> None:
        """find an area around the N, a cone with angle equal to the
        angle HN1-N-HN2"""
        num_processes: int = multip.cpu_count() // 2
        chunk_size: int = len(self.unprot_aptes_ind) // num_processes
        chunks = [self.unprot_aptes_ind[i:i+chunk_size] for i in
                  range(0, len(self.unprot_aptes_ind), chunk_size)]
        with multip.Pool(processes=num_processes) as pool:
            pool.starmap(self.process_ind, [(chunk,) for chunk in chunks])

    def process_ind(self,
                    chunk: list[int]  # Chunk of the indices for loop over
                    ) -> None:
        """doing the calculations"""
        df_i: pd.DataFrame  # Local df for each residue index
        df_nh: pd.DataFrame  # Local df for each residue index
        v_nh1: np.ndarray  # Vector from N to H1
        v_nh2: np.ndarray  # Vector from N to H2
        v_mean: np.float64  # Mean length of the NH bonds
        nh_angle: np.float64  # Angle between the two NH bonds, in radians
        atoms_around_n: pd.DataFrame  # Atoms in a radius of N
        for ind in chunk:
            df_i = self.unproton_aptes[self.unproton_aptes['mol'] == ind]
            df_nh = df_i[df_i['atom_name'].isin(['N', 'HN1', 'HN2'])]
            v_nh1, v_nh2 = self.__get_vectors(df_nh)
            v_mean, nh_angle = self.__get_hbond_len_angle(v_nh1, v_nh2)
            atoms_around_n = self.__get_atoms_around_n(df_nh, v_mean)
            self.__get_possible_posotion(v_nh1, v_nh2, v_mean)

    def __get_possible_posotion(self,
                                v_nh1: np.ndarray,  # Vector from N to H1
                                v_nh2: np.ndarray,  # Vector from N to H2
                                v_mean: np.float64,  # Length of N-H bond
                                num_samples: int = 100  # How many points
                                ) -> None:
        """find all the points which have the angle and distance
        conditions.
        """
        # Normalize the vectors
        v1_norm: np.ndarray = v_nh1 / np.linalg.norm(v_nh1)
        v2_norm: np.ndarray = v_nh2 / np.linalg.norm(v_nh2)

        # Compute the axis of rotation
        axis: np.ndarray = np.cross(v1_norm, v2_norm)
        # Compute the angle increment
        increment: float = 2 * np.pi / num_samples

        # Initialize the list of vectors
        vectors: list[np.ndarray] = []

        # Generate the vectors
        for i in range(num_samples):
            angle = i * increment
            rotated_vector = self.rotate_vector(v1_norm, axis, angle)
            normalized_vector = rotated_vector / np.linalg.norm(rotated_vector)
            vector_with_length_d = normalized_vector * v_mean
            vectors.append(vector_with_length_d)
        return vectors

    @staticmethod
    def rotate_vector(vector: np.ndarray,  # Normelized vector of one N-H
                      axis: np.ndarray,  # Direction of the axis to rotate
                      angle: float  # Angle for rotation
                      ) -> list[np.ndarray]:
        """return a list of all possible rotation"""
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
                             v_mean: np.float64  # Average length of N-H bonds
                             ) -> pd.DataFrame:
        """Generate a dataframe that includes atoms within a specified
        radius of N atoms. This is a more efficient way to check for
        overlaps with new H since it only considers the relevant atoms.
        The analysis is limited to the atoms in the box enclosing the
        NP, which contains significantly fewer atoms than the entire
        system."""
        # Extract the x, y, z coordinates from the dataframe
        coordinates: np.ndarray = \
            self.residues_atoms['box'][['x', 'y', 'z']].values
        # Build the KD-tree
        tree = KDTree(coordinates)
        # Getting the position of the N atom
        n_pos = [float(df_nh[df_nh['atom_name'] == 'N']['x']),
                 float(df_nh[df_nh['atom_name'] == 'N']['y']),
                 float(df_nh[df_nh['atom_name'] == 'N']['z'])]

        radius: float = float(v_mean) + 2
        # Find the indices of points within the radius
        indices: list[int] = tree.query_ball_point(n_pos, radius)
        # Return the points within the radius
        return self.residues_atoms['box'].iloc[indices]

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


if __name__ == '__main__':
    FindHPosition(sys.argv[1])
