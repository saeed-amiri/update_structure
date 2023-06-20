"""The purpose of this script is to divide the data file and extract
the relevant section of data. It creates separate data frames for
different residues or groups of residues. The data is accessed through
pdb_todf.py."""


import sys
import pandas as pd
import pdb_to_df as pdbf


class ProcessData:
    """process dataframe of the structure and plit them"""
    def __init__(self,
                 fname: str  # Name of the pdb file
                 ) -> None:
        self.atoms: pd.DataFrame = pdbf.Pdb(fname).atoms
        self.process_data()

    def process_data(self) -> None:
        """do it"""
        residues: list[str] = self.__get_residues()

    def __get_residues(self) -> list[str]:
        """get the list of the residues in the system"""
        residues: list[str]   # Name of the residues
        residues = list(set(self.atoms['residue_name']))
        return residues


if __name__ == '__main__':
    data = ProcessData(sys.argv[1])
