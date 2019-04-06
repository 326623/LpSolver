#!/usr/bin/python
#----------------------------
# Config section
#----------------------------
#Set the list of programs to be run
program = "popmat"
#Define the directory containing the programs to be run (relative to this script)
programsDir = "./"
#Define the number of executions for each program with each matrix
iterations = 1000
#Name of the output file
outFileDir = "./matrices/"
#-----------------------------------------------
# Do not edit beyond this line
#-----------------------------------------------
from popen2 import popen4
import sys


def main():
    m = 2
    n = 2
    for iteration in range(iterations):
        cmdLine = programsDir + program + " " + outFileDir + sys.argv[1] + str(
            iteration) + ".in " + str(m) + " " + str(n)
        print("Execution of program: " + cmdLine)
        execOut, execIn = popen4(cmdLine)
        output = execOut.read()
        m = m + 2
        n = n + 2
        print("Done")


#--------------------------------------------------
#Execution starts here
#--------------------------------------------------
if __name__ == "__main__":
    main()
