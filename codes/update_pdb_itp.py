"""This program examines the pdb file to verify if the NP is positioned
below the interface. If there are any unprotonated (protonated) APTES
molecules in the water (oil) phase, it will modify the APTES chain by
adding (removing) an H atom to the system. It may also change the names
of the remaining particles in the chain as required and, most
importantly, adjust the charges of all the particles in the chain.
Depending on the alterations in the system, it will update the number
of counterions to ensure that the system stays neutral. To operate
correctly, the program requires the interface located at the beginning
of the simulation for comparison purposes and the ability to change
the NP's location.
"""


import sys
import pandas as pd
import logger
import update_residues_pdb
import update_residues_gro
import write_pdb_file
import update_itp
import write_itp_file
import write_gro_file
import update_topo


STYLE: str = 'GRO'
LOG = logger.setup_logger('update.log')
if __name__ == '__main__':
    if STYLE == 'GRO':
        gro_data = \
            update_residues_gro.UpdateResidues(sys.argv[1], log=LOG)
        write_gro_file.write_gromacs_gro(gro_data.combine_residues,
                                         'updated_system.gro',
                                         pbc_box=gro_data.pbc_box,
                                         title=gro_data.title)
        new_hn3_dict: dict[str, pd.DataFrame] = gro_data.new_hn3
    else:
        sys.exit('\nDEPRECTED! CANNOT READ THIS! USE GRO FILES\n')
        up_data = update_residues_pdb.UpdateResidues(sys.argv[1])
        write_pdb_file.WritePdb(up_data)
        new_hn3 = up_data.new_hn3
    itp = update_itp.WrapperUpdateItp(
        param=gro_data.param, hn3_dict=new_hn3_dict)
    write_itp_file.WrapperWriteItp(itp, log=LOG)
    update_topo.ReadTop(gro_data.nr_atoms_residues, gro_data.param, log=LOG)
