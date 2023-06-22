"""In order to incorporate counterions into the system, the script
searches for unoccupied spaces within the water section and identifies
all present atoms. It ensures that the placement of counterions does
not overlap with any existing atoms based on the number of new
protonation."""


import sys
import numpy as np
import pandas as pd
import protonating as proton


class IonizationSol(proton.FindHPosition):
    """ionizing the water phase of the box. The number is equal to the
    number of deprotonation of the APTES"""
    def __init__(self,
                 fname: str  # Name of the pdb file
                 ) -> None:
        super().__init__(fname)
        self.mk_ionization()

    def mk_ionization(self) -> None:
        """get the numbers of ions and their locations in the water
        phase"""
        x_dims: np.ndarray  # Dimensions of the sol box in x
        y_dims: np.ndarray  # Dimensions of the sol box in y
        z_dims: np.ndarray  # Dimensions of the sol box in z
        sol_atoms: pd.DataFrame = self.__get_sol_phase_atoms()
        x_dims, y_dims, z_dims = self.__get_box_size(sol_atoms)
        self.__get_chunk_interval(x_dims, y_dims, z_dims)

    def __get_chunk_interval(self,
                             x_dims: np.ndarray,  # Dimensions of sol box in x
                             y_dims: np.ndarray,  # Dimensions of sol box in y
                             z_dims: np.ndarray  # Dimensions of sol box in z
                             ) -> None:
        """get dimension of each chunk-box"""
        x_lim: float = x_dims[1] - x_dims[0]
        y_lim: float = y_dims[1] - y_dims[0]
        z_lim: float = z_dims[1] - z_dims[0]
        chunk_num: np.float64 = np.floor(np.cbrt(len(self.unprot_aptes_ind)))
        x_chunks: list[tuple[float, float]] = \
            self.__get_axis_chunk(x_dims, x_lim, chunk_num)  # Chunks intervals
        y_chunks: list[tuple[float, float]] = \
            self.__get_axis_chunk(y_dims, y_lim, chunk_num)  # Chunks intervals
        z_chunks: list[tuple[float, float]] = \
            self.__get_axis_chunk(z_dims, z_lim, chunk_num)  # Chunks intervals
        print(len(x_chunks), len(y_chunks), len(z_chunks))

    @staticmethod
    def __get_axis_chunk(dims: np.ndarray,  # Dimensions of the axis
                         lims: float,  # Limites of the box
                         chunk_num: np.float64  # Number of chunks
                         ) -> list[tuple[float, float]]:
        """return the chunks of the aixs"""
        chunk_intervals: list[tuple[float, float]] = []
        for i in range(int(chunk_num)):
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
