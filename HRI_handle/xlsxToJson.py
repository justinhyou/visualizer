import math
import random
import csv
import GetDataFromURL as get
import json
import xlrd

def readData(reader):
    allEntries = []
    for row in reader:
        allEntries+=[(row['RANK_CD'], row['BUSINESS_LAST_NAME'])]
#        allEntries+=[row]
    print allEntries
#    print allEntries[0].
    return allEntries

def main():
    """ For actual data:
    focus on the columns of interest
    recreate each of the "objects" under an array
    create the json "txt"
        -go through one person
            -check if they are supervisor for each entry
            -if so, then put such "objects" under this person's "children"
                -check the current growing text file to see if this "object" already is added
    convert final string txt to json file (?)

    """

    #open reader
    in_file = open('hri.csv', "rb")
    reader = csv.DictReader(in_file)
    allData = readData(reader)

    #open writer
    out_file = open('new.json', "w")
    json.dump("asdf", out_file)


    #close out the files
    out_file.close()
    in_file.close()


main()
