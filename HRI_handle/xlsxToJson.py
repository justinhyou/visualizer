import math
import random
import csv
import GetDataFromURL as get
import json

def readData(reader):
    allEntries = []
    for row in reader:
        allEntries+=[row]
    return allEntries

def main():
    # open reader
    in_file = open('hri.csv', "rb")
    reader = csv.DictReader(in_file)
    allData = readData(reader)

    # open writer
    out_file = open('new.json', "w")
    json.dump(allData, out_file)

    # close out the files
    out_file.close()
    in_file.close()

"""creating actual JSON format"""
def main2():
        in_file = open('hri.csv', "rb")
        reader = csv.DictReader(in_file)

        out_file = open('new.json', 'w')

        for row in reader:
            json.dump(row, out_file)
            out_file.write("\n")

        out_file.close()


#main()
main2()
