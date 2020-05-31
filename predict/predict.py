import os
import sys
import csv

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, '../preprocessing'))
from remove_id import RemoveId
from pipeline_1 import Pipeline


input_path = os.path.join(file_path, '../data/test_data.txt')
pipeline_path = os.path.join(file_path, '../data/test_data_noid.txt')
prep_path = os.path.join(file_path, '../data/test_data_prep.txt')
predict_path = os.path.join(file_path, 'test_pred.csv')

# remove id
ri = RemoveId()
ri.set_paths(input_path, pipeline_path)
ri.run()

# run pipeline1
prep = Pipeline()
prep.process([pipeline_path], [prep_path])


# load model
model = tf.keras.models.load_model(
    filepath, custom_objects=None, compile=True
)


# predict
pred_sent = []
with open(prep_path, mode='r') as input:
    for line in input:
        pred_sent.append(line)


pred = model.predict(pred_sent)


# save
w = csv.writer(open(predict_path, 'w'))
w.writerow(['Id', 'Prediction'])
for i in range(len(pred)):
    w.writerow([i, [-1, 1][pred[i]]])
