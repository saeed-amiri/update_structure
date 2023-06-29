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
import update_residues
import write_pdb_file
import update_itp
import write_itp_file


if __name__ == '__main__':
    up_data = update_residues.UpdateResidues(sys.argv[1])
    write_pdb_file.WritePdb(up_data)
    itp = update_itp.UpdateItp(fname='APT_COR.itp', hn3=up_data.new_hn3)
    write_itp_file.WriteItp(itp, fname='APT_COR_updated.itp')
