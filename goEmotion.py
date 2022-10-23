import matplotlib.pyplot as plt
import numpy as np
import json
with open("goemotions.json")as f:
    constent=json.loads(f.read())
dicList = [line for line in constent]
content = [item[0] for item in dicList]


print(len(content))
num_negative = [item for item in dicList if 'negative' in item]
print(len(num_negative))

num_positive = [item for item in dicList if 'positive' in item]
print(len(num_positive))
968
num_neutral = [item for item in dicList if 'neutral' in item]
print(len(num_neutral))

num_ambiguous = [item for item in dicList if 'ambiguous' in item]
print(len(num_ambiguous))


sentiment = np.array([len(num_positive), len(num_negative), len(num_ambiguous), len(num_neutral)])
labels1 = np.array(['positive', 'negative', 'ambiguous', 'neutral'])

f = plt.figure()
plt.bar(labels1, sentiment)
plt.show()
f.savefig("sentiment.pdf")

