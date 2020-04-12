import os
import sys
import csv

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, '../preprocessing'))
from remove_id import RemoveId
from pipeline_1 import Pipeline1


input_path = os.path.join(file_path, '../data/test_data.txt')
pipeline_path = os.path.join(file_path, '../data/test_data_noid.txt')
output_path = os.path.join(file_path, '../data/test_data_prep.txt')

# remove id
ri = RemoveId()
ri.set_paths(input_path, pipeline_path)
ri.run()

# run pipeline1
prep = Pipeline1()
prep.process([pipeline_path], [output_path])


# load model


# predict


# save
'''
w = csv.writer(open(output_path, 'w'))
w.writerow(['Id', 'Prediction'])
for i in range(len(pred)):
    w.writerow([i, [-1, 1][pred[i]]])
'''
