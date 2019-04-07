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
import multiprocessing
from multiprocessing import Pool

def do_cmd(item):
    iteration, m, n = item
    cmdLine = programsDir + program + " " + outFileDir + sys.argv[1] + str(
        iteration) + ".in " + str(m) + " " + str(n)
    print("Execution of program: " + cmdLine)
    execOut, execIn = popen4(cmdLine)
    output = execOut.read()

def main():
    m = 2
    n = 2
    m_list = []
    n_list = []
    iteration_list = []
    for iteration in range(iterations):
        m_list.append(m)
        n_list.append(n)
        iteration_list.append(iteration)
        m = m + 2
        n = n + 2
    pool = Pool(multiprocessing.cpu_count())
    # pool = Pool(1)

    for _ in pool.imap(do_cmd, zip(iteration_list, m_list, n_list)):
        pass

    print("Done")
        # cmdLine = programsDir + program + " " + outFileDir + sys.argv[1] + str(
        #     iteration) + ".in " + str(m) + " " + str(n)
        # print("Execution of program: " + cmdLine)
        # execOut, execIn = popen4(cmdLine)
        # output = execOut.read()
        # m = m + 2
        # n = n + 2
        # print("Done")


#--------------------------------------------------
#Execution starts here
#--------------------------------------------------
if __name__ == "__main__":
    main()
