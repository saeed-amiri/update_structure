"""The purpose of this script is to divide the data file and extract
the relevant section of data. It creates separate data frames for
different residues or groups of residues. The data is accessed through
pdb_todf.py."""


import sys
import multiprocessing as multip
import numpy as np
import pandas as pd
import pdb_to_df as pdbf
import read_param as param
import logger


class ProcessData:
    """process dataframe of the structure and plit them"""
    def __init__(self,
                 fname: str  # Name of the pdb file
                 ) -> None:
        self.param: dict[str, float]  # All the parameters from input
        self.param = param.ReadParam(
            log=logger.setup_logger('read_param.log')).param
        self.atoms: pd.DataFrame = pdbf.Pdb(fname).atoms
        self.residues_atoms: dict[str, pd.DataFrame]  # Atoms info for each res
        self.residues_atoms = self.__get_atoms()
        # All the unprtonated aptes to be protonated:
        self.unproton_aptes: pd.DataFrame  # Atoms info
        self.unprot_aptes_ind: list[int]  # Index of the APTES
        self.unproton_aptes, self.unprot_aptes_ind = self.process_data()

    def process_data(self) -> tuple[np.ndarray, list[int]]:
        """check and finds the unprotonated aptes group which has N at
        interface"""
        zrange: tuple[float, float]  # Lower and upper bound of interface
        zrange = self.__get_interface_range()
        interface_aptes: list[int]  # Indices of all the APTES at interface
        interface_aptes = self.__get_aptes_indices(zrange)
        unprot_aptes_ind: list[int]  # Index of the APTES to be protonated
        unprot_aptes_ind = self.__get_unprto_chain(interface_aptes)
        return self.get_aptes_unproto(unprot_aptes_ind), unprot_aptes_ind

    def get_aptes_unproto(self,
                          unprot_aptes_ind: list[int]  # Index of the APTES
                          ) -> pd.DataFrame:
        """get all atoms in the chains of the unprotonated APTES"""
        df_apt: pd.DataFrame = self.residues_atoms['APT']
        return df_apt[df_apt['mol'].isin(unprot_aptes_ind)]

    def __get_unprto_chain(self,
                           interface_aptes: list[int]  # Indices of APTES
                           ) -> list[int]:
        """find all the chains at the intrface"""
        df_apt: pd.DataFrame = self.residues_atoms['APT']
        unprotonated_aptes: list[int] = []
        # Split the interface_aptes list into chunks
        num_processes: int = multip.cpu_count() // 2
        chunk_size: int = len(interface_aptes) // num_processes
        chunks = [interface_aptes[i:i+chunk_size] for i in
                  range(0, len(interface_aptes), chunk_size)]
        # Create a Pool of processes
        with multip.Pool(processes=num_processes) as pool:
            # Process the chunks in parallel
            results = pool.starmap(self.process_chunk,
                                   [(chunk, df_apt) for chunk in chunks])
        # Combine the results
        for result in results:
            unprotonated_aptes.extend(result)
        # Close the pool to release resources
        del df_apt
        return unprotonated_aptes

    @staticmethod
    def process_chunk(chunk: np.ndarray,  # Chunk of a APTES indices
                      df_apt: pd.DataFrame  # For the APTES at the interface
                      ) -> list[int]:
        """get index of unprotonated aptes"""
        unprotonated_aptes_chunk: list[int] = []  # Index of unprotonated APTES
        for ind in chunk:
            df_i = df_apt[df_apt['mol'] == ind]
            # Check if 'NH3' is present in 'atom_name'
            if not df_i['atom_name'].str.contains('NH3').any():
                unprotonated_aptes_chunk.append(ind)
        return unprotonated_aptes_chunk

    def __get_aptes_indices(self,
                            zrange: tuple[float, float]  # Bound of interface
                            ) -> list[int]:
        """get the index of all the aptes at the interface range"""
        # Filter the DataFrame based on the specified conditions
        df_apt = self.residues_atoms['APT']
        filtered_df = df_apt[(df_apt['atom_name'] == 'N') &
                             (df_apt['z'].between(zrange[0], zrange[1]))]

        # Get the 'mol' values for the filtered atoms
        del df_apt
        return filtered_df['mol'].values

    def __get_interface_range(self) -> tuple[float, float]:
        """find all the aptes at interface"""
        return (self.param['INTERFACE']-self.param['INTERFACE_WIDTH']/2+100,
                self.param['INTERFACE']+self.param['INTERFACE_WIDTH']/2+100)

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
