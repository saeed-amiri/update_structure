"""write updated GRO file"""

import pandas as pd
import update_residues_gro


def write_gromacs_gro(gro_data: pd.DataFrame,
                      filename: str,  # Name of the output file
                      pbc_box=None,
                      title=None
                      ) -> None:
    """Write DataFrame to a GROMACS gro file."""
    df_i: pd.DataFrame = gro_data.copy()
    with open(filename, 'w', encoding='utf8') as gro_file:
        if title:
            gro_file.write(f'{title}\n')  # Add a comment line
        gro_file.write(f'{len(df_i)}\n')  # Write the total number of atoms
        for _, row in df_i.iterrows():
            line = f'{row["residue_number"]:>5}' \
                   f'{row["residue_name"]:<5}' \
                   f'{row["atom_name"]:>5}' \
                   f'{row["atom_id"]:>5}' \
                   f'{row["x"]:8.3f}' \
                   f'{row["y"]:8.3f}' \
                   f'{row["z"]:8.3f}' \
                   f'{row["vx"]:8.4f}' \
                   f'{row["vy"]:8.4f}' \
                   f'{row["vz"]:8.4f}\n'
            gro_file.write(line)
        if pbc_box:
            gro_file.write(f'{pbc_box}\n')
