#!/usr/bin/python
#----------------------------
# Config section
#----------------------------
#Set the list of programs to be run
programs = ["cpu_solver", "gpu_solver"]
#Define the directory containing the programs to be run (relative to this script)
programsDir = "./"
#Define the number of executions for each program with each matrix
iterations = 1
#Name of the output file
outFileName = "results.csv"
dataFileName = "results.dat"
innerFileName = "inner_results.dat"
#-----------------------------------------------
# Do not edit beyond this line
#-----------------------------------------------
from popen2 import popen4
import re
import sys

epsilon = 1e-20

def getFileList(wildcardedFileNames):
    """Gets the list of files from command line. If some
    filenames contain
    wildcards, they are divided in single file names"""
    import glob
    fileList = []
    """Each fileName from the command line can contain
    wildcards, therefore it may be a list of files.
    This two "for" extract the list of files in every
    group and adds the files, one by one, to the list
    of file names."""
    for fileName in wildcardedFileNames:
        tmpList = glob.glob(fileName)
        for oneFile in tmpList:
            fileList.append(oneFile)
    return fileList


def main():
    #Get the file list
    files = sys.argv[1:]
    fileList = getFileList(files)
    #print fileList
    #Open the output file and prepare its heading
    print "Preparing the output file..."
    OUTFILE = file(outFileName, "w")
    DATAFILE = file(dataFileName, "w")
    INNERFILE = file(innerFileName, "w")
    OUTFILE.write(
        "Program; Matrix file name; # constraints; # variables; Matrix size; Elapsed time[ns]; Optimum\n"
    )
    DATAFILE.write(
        "# constraints; # variables; Matrix size; lpsolver time[ns]; culpsolver time[ns]; Speedup\n"
    )
    INNERFILE.write(
        "# constraints; # variables; Matrix size; lpsolver ev_time[ns]; lpsolver lv_time[ns]; lpsolver b_time[ns]; culpsolver ev_time[ns];culpsolver lv_time[ns]; culpsolver b_time[ns];ev_speedup;lv_speedup; b_speedup\n"
    )

    print "Done"
    #Prepare the extraction regexp
    varRE = re.compile("m=(\d+) n=(\d+)")
    sizeRE = re.compile("Size: (\d+)")
    elapsedRE = re.compile("Elapsed time: (\d+.\d+)")
    optRE = re.compile("Optimum found: (\d+.\d+)")
    evRE = re.compile("Entering variable computation time: (\d+.\d+)")
    lvRE = re.compile("Leaving variable computation time: (\d+.\d+)")
    bRE = re.compile("Binv updating time: (\d+.\d+)")
    noptRE = re.compile("^Problem")
    for fileName in fileList:
        for iteration in range(iterations):
            opt = []
            times = []
            evTimes = []
            lvTimes = []
            bTimes = []
            for p in [0, 1]:
                cmdLine = programsDir + programs[p] + " " + fileName
                print("Execution #" + str(iteration + 1) + " of program: " +
                      programs[p] + " with matrix in " + fileName)
                execOUT, execIN = popen4(cmdLine)
                print "Waiting for the results of the determinant calculation"
                output = execOUT.read()
                print "Extracting informations from the output"
                #Extract
                numCons = varRE.search(output).group(1)
                numVar = varRE.search(output).group(2)
                size = sizeRE.search(output).group(1)
                times.append(elapsedRE.search(output).group(1))
                evTimes.append(evRE.search(output).group(1))
                lvTimes.append(lvRE.search(output).group(1))
                bres = bRE.search(output)
                if type(bres) == type(noptRE.search("Problem")):
                    bTimes.append(bres.group(1))
                else:
                    bTimes.append("NaN")
                res = optRE.search(output)
                if type(res) == type(noptRE.search("Problem")):
                    opt.append(res.group(1))
                else:
                    opt.append("NaN")
                if p == 0:
                    DATAFILE.write(
                        str(numCons) + "\t" + str(numVar) + "\t" + str(size) +
                        "\t" + str(times[0]) + "\t")
                    INNERFILE.write(
                        str(numCons) + "\t" + str(numVar) + "\t" + str(size) +
                        "\t" + str(evTimes[0]) + "\t" + str(lvTimes[0]) +
                        "\t" + str(bTimes[0]) + "\t")
                else:
                    # divide by zero error, add a small fraction onto it
                    DATAFILE.write(
                        str(times[1]) + "\t" +
                        str(float(times[0]) / (float(times[1]) + epsilon)) + "\n")
                    INNERFILE.write(
                        str(evTimes[1]) + "\t" + str(lvTimes[1]) + "\t" +
                        str(bTimes[1]) + "\t" +
                        str(float(evTimes[0]) / (float(evTimes[1]) + epsilon)) + "\t" +
                        str(float(lvTimes[0]) / (float(lvTimes[1]) + epsilon)) + "\t" +
                        str(float(bTimes[0]) / (float(bTimes[1]) + epsilon)) + "\n")
                OUTFILE.write(programs[p] + ";" + fileName + ";" +
                              str(numCons) + ";" + str(numVar) + ";" +
                              str(size) + ";" + str(times[p]) + ";" +
                              str(opt[p]) + "\n")
            if opt[0] == opt[1]:
                OUTFILE.write("Fitting: OK\n")
            else:
                OUTFILE.write("Fitting: KO\n")

    print "Done"
    print "Data saved in " + dataFileName
    print "Summary saved in " + outFileName
    print "Inner data saved in " + innerFileName
    OUTFILE.close()
    DATAFILE.close()
    INNERFILE.close()


#--------------------------------------------------
#Execution starts here
#--------------------------------------------------
if __name__ == "__main__":
    main()
