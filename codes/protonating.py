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
        for ind in chunk:
            df_i = self.unproton_aptes[self.unproton_aptes['mol'] == ind]
            df_nh = df_i[df_i['atom_name'].isin(['N', 'HN1', 'HN2'])]
            self.__get_vectors(df_nh)

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
