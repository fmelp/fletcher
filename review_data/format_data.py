import json
import csv
import numpy as np

labels = ['source', 'review', 'sentiment', 'prediction', 'id']

with open('all_fake_neg.txt', 'r') as f:
    fake_neg = f.readlines()
fake_neg_final = []
for i, review in enumerate(fake_neg):
    fake_neg_final.append(['mturk', review.strip(), 0, 0, 'murk_n%i' %i])

with open('all_fake_pos.txt', 'r') as f:
    fake_pos = f.readlines()
fake_pos_final = []
for i, review in enumerate(fake_pos):
    fake_neg_final.append(['mturk', review.strip(), 1, 0, 'murk_p%i' %i])

with open('all_unlabeled_neg.txt', 'r') as f:
    real_neg = f.readlines()
real_neg_final = []
for i, review in enumerate(real_neg):
    real_neg_final.append(['trip_advisor', review.strip(), 0, np.nan, 'ta_n%i' %i])

with open('all_unlabeled_pos.txt', 'r') as f:
    real_pos = f.readlines()
real_pos_final = []
for i, review in enumerate(real_pos):
    real_pos_final.append(['trip_advisor', review.strip(), 1, np.nan, 'ta_p%i' %i])


with open('expedia_data3.json') as f:
    expedia = json.load(f)

ta = []
with open('ta_new3.csv', 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if "Room Tip:" in row[0]:
            review = row[0].split("Room Tip:")
            review = review[0].strip()
            ta.append(['trip_advisor', review, np.nan, np.nan, 'ta_s%i' %i])
        elif "More" in row[0]:
            continue
        else:
            review = row[0].strip()
            ta.append(['trip_advisor', review, np.nan, np.nan, 'ta_s%i' %i])


# ta = []
# with open('ta_new3.csv', 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         ta.append(row)




expedia_final = []
counter = 0
for i, key in enumerate(expedia.keys()):
    if i != 0:
        counter += 1
    for j, review in enumerate(expedia[key]):
        if j != 0:
            counter += 1
        expedia_final.append(['expedia', review.strip(), np.nan, 1, 'ex%i' %counter])


with open('review_data_ids.csv', 'w') as fout:
    writer = csv.writer(fout)
    writer.writerow(labels)
    for row in fake_neg_final:
        writer.writerow(row)
    for row in fake_pos_final:
        writer.writerow(row)
    for row in real_neg_final:
        writer.writerow(row)
    for row in real_pos_final:
        writer.writerow(row)
    for row in expedia_final:
        writer.writerow(row)
    for row in ta:
        writer.writerow(row)
