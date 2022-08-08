import json
import sys

with open(sys.argv[1], "r") as f:
    dialog = [json.loads(line) for line in f]

for d in dialog:
    for index in range(1, len(d["dialog"]), 2):
        sentence = d["dialog"][index]
        end = max(sentence.rfind('?', 0, len(sentence)), sentence.rfind('!', 0, len(sentence)), sentence.rfind('.', 0, len(sentence)))
        d["dialog"][index] = sentence[:end+1]

with open(sys.argv[2], "w") as f:
    for idx, d in enumerate(dialog):
        f.write(json.dumps(d) + "\n")