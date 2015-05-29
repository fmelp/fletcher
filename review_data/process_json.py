import json

with open('expedia_data2.json') as f:
    data = json.load(f)

new_keys = []
for key in data.values():
    new_key = []
    for i, review in enumerate(key):
        try:
            str(key[i])
            new_key.append(key[i])
        except UnicodeError:
            continue
    new_keys.append(new_key)

new = dict(zip(data.keys(), new_keys))

print str(new_keys[0][9])

with open('expedia_data3.json', 'w') as fout:
    json.dump(new, fout, indent=2)
