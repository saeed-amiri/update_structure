"""reading GRO files:
The GRO file format is a plain-text file format commonly used in
molecular dynamics simulations, including those performed with GROMACS.
It represents the coordinates and other information of atoms in a
system at a specific time step. Here is the general format of a GRO
file:

The first two lines of the file contain header information:

    Line 1: Title line, usually a brief description of the system or
    simulation.
    Line 2: Number of atoms in the system.

Following the header, each atom in the system is represented by one or
more lines in the file. The format for each atom line is as follows:

    Columns 1-5: Atom serial number.
    Columns 6-10: Atom name or type.
    Columns 11-15: Residue name.
    Columns 16-20: Residue number.
    Columns 21-28: Atom coordinates along the x-axis (in nanometers).
    Columns 29-36: Atom coordinates along the y-axis (in nanometers).
    Columns 37-44: Atom coordinates along the z-axis (in nanometers).
    Columns 45-48: Atom velocity (optional; not always present in GRO
    files).
    Note: The coordinates and velocities are typically given in
    nanometers.

The atom lines are repeated for each atom in the system, usually in
consecutive order. At the end of the file, there is an additional line
that specifies the box size or periodic boundary conditions of the
simulation box."""


