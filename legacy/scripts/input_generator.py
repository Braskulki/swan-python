swan = (
"PROJECT 'TEST' '001'\n"
"\n"
"MODE STATIONARY TWODIMENSIONAL\n"
"\n"
"CGRID REGULAR -48 -25 0 4 3 50 40\n"
"\n"
"INPGRID BOTTOM REGULAR -48 -25 0 4 3 50 40\n"
"READINP BOTTOM 1 'D:/repositories/swan/data/processed/depth.bot'\n"
"\n"
"WIND 10 0\n"
"\n"
"GEN3\n"
"CIRCLE 36 0.04 1.0\n"
"\n"
"BLOCK 'COMPGRID' 'output.dat' HSIGN DIR TPS\n"
"\n"
"COMPUTE\n"
"STOP\n"
)

with open("input.swn", "w", newline="\n") as f:
    f.write(swan)