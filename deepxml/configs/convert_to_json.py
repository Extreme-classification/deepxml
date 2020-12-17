import json
import csv 
import sys 

input_filename = sys.argv[1]
output_filename = "{}.json".format(input_filename[:-3])

input_file = open(input_filename)
output_file = open(output_filename, 'w')

reader = csv.reader(input_file, delimiter="=")
data_dict = {}

for row in reader:
    if len(row) == 0:
        continue
    
    a, b = row

    try: 
        data_dict[a] = int(b)
    except:
        try:
            data_dict[a] = float(b)
        except:
            if a == "order":
                data_dict[a] = [x[1:-1] for x in b[1:-1].split()]
            elif b[0] == "(":
                data_dict[a] = float(b[1:-1])
            else:
                data_dict[a] = b[1:-1]

json.dump(data_dict, output_file, indent=4)