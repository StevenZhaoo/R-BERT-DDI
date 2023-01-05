# Author: StevenChaoo
# -*- coding:UTF-8 -*-


import os

train_file = open("train.txt", "w")
dev_file = open("dev.txt", "w")
test_file = open("test.txt", "w")

num = 0

for _, _, files in os.walk("/home/ZhengyiZhao/File/DDI_Data_Create/DDIbrat/"):
    for single_file in files:
        raw_file = open("/home/ZhengyiZhao/File/DDI_Data_Create/DDIbrat/{}.txt".format(single_file[:-4]), "r")
        sent_start = 0
        sent_end = 0
        positions = []
        sentences = []
        for idx, raw_line in enumerate(raw_file.readlines()):
            sentences.append(raw_line.strip())
            if idx == 0:
                sent_start = sent_start
                sent_end = len(raw_line.strip()) - 1
            else:
                sent_start = sent_end + 2
                sent_end = sent_start + len(raw_line.strip()) - 1
            positions.append((sent_start, sent_end))

        new_sentences = []
        for sent_idx, sentence in enumerate(sentences):
            ann_file = open("/home/ZhengyiZhao/File/DDI_Data_Create/DDIbrat/{}.ann".format(single_file[:-4]), "r")
            ids = []
            entities = []
            relations = []
            for ent_line in ann_file.readlines():
                ent_line_list = ent_line.strip().split("\t")
                if len(ent_line_list) == 3:
                    front = int(ent_line_list[1].split(" ")[1]) >= positions[sent_idx][0]
                    behind = int(ent_line_list[1].split(" ")[-1])-1 <= positions[sent_idx][1]
                    if front and behind:
                        ent_info = {}
                        ids.append(ent_line_list[0])
                        ent_info["ent_id"] = ent_line_list[0]
                        ent_info["ent_type"] = ent_line_list[1].split(" ")[0]
                        ent_info["ent_start"] = int(ent_line_list[1].split(" ")[1])-positions[sent_idx][0]
                        ent_info["ent_end"] = int(ent_line_list[1].split(" ")[-1])-positions[sent_idx][0]
                        ent_info["ent_text"] = ent_line_list[2]
                        entities.append(ent_info)
                else:
                    if ent_line_list[1].split(" ")[1][5:] in ids:
                        rel_info = {}
                        rel_info["rel_id"] = ent_line_list[0]
                        rel_info["type"] = ent_line_list[1].split(" ")[0]
                        rel_info["sub"] = ent_line_list[1].split(" ")[1][5:]
                        rel_info["obj"] = ent_line_list[1].split(" ")[2][5:]
                        relations.append(rel_info)

            if len(entities) > 0:
                for ent_idx, entity in enumerate(entities):
                    if ent_idx != len(entities)-1:
                        for temp in range(len(entities)-ent_idx):
                            if temp != 0:
                                for rel in relations:
                                    if entity["ent_id"] == rel["sub"] and entities[ent_idx+temp]["ent_id"] == rel["obj"]:
                                        new_sentence = "{}\t{} <e1:{}> {} </e1:{}> {} <e2:{}> {} </e2:{}> {}".format(
                                            rel["type"],
                                            sentence[:entity["ent_start"]],
                                            entity["ent_type"],
                                            entity["ent_text"],
                                            entity["ent_type"],
                                            sentence[entity["ent_end"]:entities[ent_idx+temp]["ent_start"]],
                                            entities[ent_idx+temp]["ent_type"],
                                            entities[ent_idx+temp]["ent_text"],
                                            entities[ent_idx+temp]["ent_type"],
                                            sentence[entities[ent_idx+temp]["ent_end"]:])
                                        break
                                    else:
                                        new_sentence = "negative\t{} <e1:{}> {} </e1:{}> {} <e2:{}> {} </e2:{}> {}".format(
                                            sentence[:entity["ent_start"]],
                                            entity["ent_type"],
                                            entity["ent_text"],
                                            entity["ent_type"],
                                            sentence[entity["ent_end"]:entities[ent_idx+temp]["ent_start"]],
                                            entities[ent_idx+temp]["ent_type"],
                                            entities[ent_idx+temp]["ent_text"],
                                            entities[ent_idx+temp]["ent_type"],
                                            sentence[entities[ent_idx+temp]["ent_end"]:])
                                new_sentences.append(new_sentence)
            ann_file.close()
        raw_file.close()
        for sent in new_sentences:
            if sent != "":
                num += 1
                if num <= 60000:
                    train_file.write(sent)
                    train_file.write("\n")
                elif num <= 68338 and num > 60000:
                    dev_file.write(sent)
                    dev_file.write("\n")
                else:
                    test_file.write(sent)
                    test_file.write("\n")
