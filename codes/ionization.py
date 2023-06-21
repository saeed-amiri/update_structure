"""In order to incorporate counterions into the system, the script
searches for unoccupied spaces within the water section and identifies
all present atoms. It ensures that the placement of counterions does
not overlap with any existing atoms based on the number of new
protonation."""


import sys
import protonating as proton


if __name__ == '__main__':
    prtotnation = proton.FindHPosition(sys.argv[1])
