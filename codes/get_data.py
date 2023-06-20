"""The purpose of this script is to divide the data file and extract
the relevant section of data. It creates separate data frames for
different residues or groups of residues. The data is accessed through
pdb_todf.py."""


import sys
import pandas as pd
import pdb_to_df as pdbf
import read_param as param
import logger


class ProcessData:
    """process dataframe of the structure and plit them"""
    def __init__(self,
                 fname: str  # Name of the pdb file
                 ) -> None:
        self.atoms: pd.DataFrame = pdbf.Pdb(fname).atoms
        self.param: dict[str, float]  # All the parameters from input
        self.param = param.ReadParam(
            log=logger.setup_logger('read_param.log')).param
        self.residues_atoms: dict[str, pd.DataFrame]  # Atoms info for each res
        self.residues_atoms = self.__get_atoms()
        self.process_data()

    def process_data(self):
        """process here"""
        self.__check_aptes()

    def __check_aptes(self) -> None:
        """check and finds the unprotonated aptes group which has N at
        interface"""
        zrange: tuple[float, float]  # Lower and upper bound of interface
        zrange = self.__get_interface_range()

    def __get_interface_range(self) -> tuple[float, float]:
        """find all the aptes at interface"""
        return (self.param['INTERFACE']-self.param['INTERFACE_WIDTH']/2,
                self.param['INTERFACE']+self.param['INTERFACE_WIDTH']/2)

    def __get_atoms(self) -> dict[str, pd.DataFrame]:
        """get all the atoms for each residue"""
        residues: list[str] = self.__get_residues_names()
        residues_atoms: dict[str, pd.DataFrame] = \
            self.__get_residues_atoms(residues)
        return residues_atoms

    def __get_residues_atoms(self,
                             residues: list[str]  # Name of the residues
                             ) -> dict[str, pd.DataFrame]:
        """return a dictionary of all the residues with thier atoms
        information"""
        residues_atoms: dict[str, pd.DataFrame] = {}  # All the atoms data
        for res in residues:
            residues_atoms[res] = self.atoms[self.atoms['residue_name'] == res]
        return residues_atoms

    def __get_residues_names(self) -> list[str]:
        """get the list of the residues in the system"""
        residues: list[str]   # Name of the residues
        residues = list(set(self.atoms['residue_name']))
        return residues


if __name__ == '__main__':
    data = ProcessData(sys.argv[1])
