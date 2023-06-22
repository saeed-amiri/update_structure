"""In order to incorporate counterions into the system, the script
searches for unoccupied spaces within the water section and identifies
all present atoms. It ensures that the placement of counterions does
not overlap with any existing atoms based on the number of new
protonation."""


import sys
import multiprocessing as multip
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
        x_chunks: list[tuple[float, float]]
        y_chunks: list[tuple[float, float]]
        z_chunks: list[tuple[float, float]]
        sol_atoms: pd.DataFrame = self.__get_sol_phase_atoms()
        x_dims, y_dims, z_dims = self.__get_box_size(sol_atoms)
        x_chunks, y_chunks, z_chunks = \
            self.__get_chunk_interval(x_dims, y_dims, z_dims)
        self.__get_chunk_atoms(sol_atoms, x_chunks, y_chunks, z_chunks)

    def __get_chunk_atoms(self,
                          sol_atoms: pd.DataFrame,  # All Sol phase atoms
                          x_chunks: list[tuple[float, float]],  # intervals
                          y_chunks: list[tuple[float, float]],  # intervals
                          z_chunks: list[tuple[float, float]],  # intervals
                          ) -> None:
        """get atoms within each chunk box"""

        chunks = [(sol_atoms, x_i, y_i, z_i)
                  for x_i in x_chunks
                  for y_i in y_chunks
                  for z_i in z_chunks]

        with multip.Pool() as pool:
            df_list = pool.starmap(self._process_chunk_box,
                                   [chunk for chunk in chunks])

    @staticmethod
    def _process_chunk_box(sol_atoms: pd.DataFrame,  # All Sol phase atoms
                           x_i: np.ndarray,  # interval for the box
                           y_i: np.ndarray,  # interval for the box
                           z_i: np.ndarray  # interval for the box
                           ) -> pd.DataFrame:
        """process the chunk box"""
        df_i = sol_atoms[(sol_atoms['x'] >= x_i[0]) &
                         (sol_atoms['x'] < x_i[1]) &
                         (sol_atoms['y'] >= y_i[0]) &
                         (sol_atoms['y'] < y_i[1]) &
                         (sol_atoms['z'] >= z_i[0]) &
                         (sol_atoms['z'] < z_i[1])
                         ]
        return df_i

    def __get_chunk_interval(self,
                             x_dims: np.ndarray,  # Dimensions of sol box in x
                             y_dims: np.ndarray,  # Dimensions of sol box in y
                             z_dims: np.ndarray  # Dimensions of sol box in z
                             ) -> tuple[list[tuple[float, float]],
                                        list[tuple[float, float]],
                                        list[tuple[float, float]]]:
        """get dimension of each chunk-box and return the dimension of
        each chunk-box"""
        x_lim: float = x_dims[1] - x_dims[0]
        y_lim: float = y_dims[1] - y_dims[0]
        z_lim: float = z_dims[1] - z_dims[0]
        chunk_num: int = int(np.cbrt(len(self.unprot_aptes_ind)))
        chunk_axis: tuple[int, int, int] = self.__get_chunk_numbers(chunk_num)
        x_chunks: list[tuple[float, float]] = \
            self.__get_axis_chunk(x_dims, x_lim, chunk_axis[0])  # Chunks range
        y_chunks: list[tuple[float, float]] = \
            self.__get_axis_chunk(y_dims, y_lim, chunk_axis[1])  # Chunks range
        z_chunks: list[tuple[float, float]] = \
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
                         ) -> list[tuple[float, float]]:
        """return the chunks of the aixs"""
        chunk_intervals: list[tuple[float, float]] = []
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
