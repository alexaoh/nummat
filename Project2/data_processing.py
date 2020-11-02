"""Processing of the binary data generated on Markov.

Import binary files via pickle. 
Change format of all the binary files to csv (make new files).
Sort the csv files ascending in order of J (col 4: value of objective function in last iteration).
Possible to delete old, unsorted csv files in the end also. 

NB: The commands used via subprocess are only tested on some Linux distributions. 
Hence, we do not know if these commands work on Windows or Mac. They probably work
on Mac, but probably not on Windows. 
"""
import csv
import pickle
import os
import subprocess
import sys

# (relative) paths.
binpath = "data/markov/tol0point005BIN/"
csvpath = "data/markov/tol0point005CSV/"

def find_filenames(binpath, csvpath):
    """Find filenames from testing and training respectively, after finding all filenames in binpath."""
    filenames = os.listdir(binpath)
    test_filenames = []
    train_filenames = []
    for filename in filenames:
        if "test" in filename:
            test_filenames.append(filename)
        else:
            train_filenames.append(filename)
    return test_filenames, train_filenames

def convert_to_csv(*, case, filenames):
    """Convert binary files to csv. Train or test needs to be specified."""
    for filename in filenames:
        infile = open(binpath+filename, 'rb')
        config = pickle.load(infile)
        infile.close()

        fields = [ 'K', 'd', 'h', 'J']

        if case == "test":
            fields += ['Ratio']
        K_values = [10, 15, 20, 23, 30]
        d_values = [2, 3, 4]
        h_values = [0.05, 0.1, 0.2, 0.3, 0.4]

        with open(csvpath+filename+".csv", "w") as f:
            w = csv.writer(f) 
            w.writerow(fields)
            for K in K_values:
                for d in d_values:
                    for h in h_values:
                        if case == "test":
                            w.writerow([K, d, h, config[K][d][h][0], config[K][d][h][1]])
                        elif case == "train":
                            w.writerow([K, d, h, config[K][d][h]])

# Get correct filenames.
test_filenames, train_filenames = find_filenames(binpath, csvpath)

# Converts the binary files to csv. 
convert_to_csv(case = "train", filenames = train_filenames)
convert_to_csv(case = "test", filenames = test_filenames)

# Sort all the csv files in ascending order according to column 4 via Linux command.

def sort_files(*, case, filenames):
    """Output sorted csv files to new files with 'S' leading char in filename."""
    path = sys.path[0]+"/"+csvpath
    for filename in filenames:
        if case == "train":
            cmd = "(head -n1 "+path+filename+".csv && LC_NUMERIC=en_US.UTF-8 sort -t\",\" -k4 -g <(tail -n+2 "+path+filename+".csv) ) > "+path+"S"+filename+".csv"
        elif case == "test":
            cmd = "(head -n1 "+path+filename+".csv && LC_NUMERIC=en_US.UTF-8 sort -t\",\" -k4,4 -k5,5 -g <(tail -n+2 "+path+filename+".csv) ) > "+path+"S"+filename+".csv"
        print(cmd)
        process = subprocess.Popen(cmd, shell = True, stdout=subprocess.PIPE, executable="/bin/bash")
        output, error =  process.communicate()
        returncode = process.wait()

sort_files(case = "train", filenames = train_filenames)
sort_files(case = "test", filenames = test_filenames)

def delete_unsorted_files(*, case, filenames):
    """Delete the old files, after sorting. Files deleted according to old names (without 'S' in front)."""
    path = sys.path[0]+"/"+csvpath
    for filename in filenames:
        cmd = "rm -f " + path+filename +".csv"
        process = subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE)
        output, error = process.communicate()
        returncode = process.wait()

#delete_unsorted_files(case = "train", filenames = train_filenames)
#delete_unsorted_files(case = "test", filenames = test_filenames)
