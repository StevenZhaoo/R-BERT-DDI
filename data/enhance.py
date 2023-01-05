# Author: StevenChaoo
# -*- coding:UTF-8 -*-


f = open("train.tsv", "r")
enh_f = open("enh_train.tsv", "w")


for line in f.readlines():
    line_list = line.strip().split("\t")
    if line_list[0] == "int":
        for i in range(14):
            enh_f.write(line)
    elif line_list[0] == "advise":
        for i in range(2):
            enh_f.write(line)
    else:
        enh_f.write(line)
