
import os



filelist = []
countlist = []
rootdir = "/Users/bofengmessi/PycharmProjects/HelloW/static/NSWSC"
for parent,dirnames,filenames in os.walk(rootdir):
    for filename in filenames:
        filelist.append(filename)
        # print(filename)
        countlist.append("static/NSWSC/" + filename)