import json
import pprint
oowl_split_txt = "/data5/drone_machinelearning/Datasets/OOWL_split.txt"
output_json = "./OWILD.json"

owild = []
with open (oowl_split_txt, 'r') as f:
    f = f.read().split('\n')[:-1]
    curr_class = None
    curr_partition = None
    curr_dict = None
    for line in f:
        if '\t' not in line:
            read_class = ' '.join(line.split(' ')[1:])
            if 'training' in line:
                curr_partition = 'train'
            else:
                curr_partition = 'test'
            read_class = read_class[:read_class.index(curr_partition)-1]

            if read_class != curr_class:
                owild.append(curr_dict)
                curr_class = read_class
                curr_dict = {"taxonomy_id": curr_class, "taxonomy_name":curr_class, "train":[], "test":[]}
        else:
            curr_dict[curr_partition].append(line.replace('\t',''))
owild.append(curr_dict)
owild = owild[1:]
pprint.pprint(owild)

with open(output_json, 'w') as f:
    j = json.dump(owild, f, indent=4)

#f = open(output_json, 'w')
#print(j, file=f)
#f.close()

