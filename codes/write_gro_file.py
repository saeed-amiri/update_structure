"""write updated GRO file"""

import pandas as pd
import update_residues_gro


def write_gromacs_gro(gro_data: update_residues_gro.UpdateResidues,
                      filename: str  # Name of the output file
                      ) -> None:
    """Write DataFrame to a GROMACS gro file."""
    df_i: pd.DataFrame = gro_data.combine_residues
    with open(filename, 'w') as gro_file:
        gro_file.write(f'{gro_data.title}\n')  # Add a comment line
        gro_file.write(f'{len(df_i)}\n')  # Write the total number of atoms
        for _, row in df_i.iterrows():
            line = f'{row["residue_number"]:>5}' \
                   f'{row["residue_name"]:<5}' \
                   f'{row["atom_name"]:<5}' \
                   f'{row["atom_id"]:>5}' \
                   f'{row["x"]:8.3f}' \
                   f'{row["y"]:8.3f}' \
                   f'{row["z"]:8.3f}' \
                   f'{row["vx"]:8.4f}' \
                   f'{row["vy"]:8.4f}' \
                   f'{row["vz"]:8.4f}\n'
            gro_file.write(line)
        gro_file.write(f'{gro_data.pbc_box}')
