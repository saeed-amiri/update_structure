"""This script utilizes get_data.py to obtain a list and data for the
APTES chain that requires protonation. The script then identifies
suitable positions for each additional hydrogen atom near the N atom,
ensuring it does not interfere with other atoms in the surrounding
residues. Initially, the script locates a plausible area for the
hydrogen atom and subsequently checks for the presence of other atoms.
Finally, the script places the hydrogen atom in a suitable position.
"""


import sys
import numpy as np
import pandas as pd
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
        for ind in self.unprot_aptes_ind:
            df_i = self.unproton_aptes[self.unproton_aptes['mol'] == ind]
            df_nh = df_i[df_i['atom_name'].isin(['N', 'HN1', 'HN2'])]
            self.__get_vectors(df_nh)

    def __get_vectors(self,
                      df_nh: pd.DataFrame  # Contains only N and NHs
                      ) -> tuple[np.ndarray, np.ndarray]:
        """find the vectors between N and H atoms"""
        vec_dict: dict[str, np.ndarray] = {}
        vec_dict = self.__get_posixyz(df_nh)

    @staticmethod
    def __get_posixyz(df_nh: pd.DataFrame  # Contains only N and NHs
                      ) -> dict[str, np.ndarray]:
        """return position (xyz) of the atoms"""
        vec_dict: dict[str, np.ndarray] = {}
        for _, row in df_nh.iterrows():
            vec = np.array([ row['x'], row['y'], row['z']])
            vec_dict[row['atom_name']] = vec
        return vec_dict

if __name__ == '__main__':
    FindHPosition(sys.argv[1])
