import matplotlib.pyplot as plt
import numpy as np
import json


with open("goemotions.json")as f:
    dicList=json.load(f)

num_sadness = [item for item in dicList if 'sadness' in item]
print(len(num_sadness))

num_neutral2 = [item for item in dicList if 'neutral' in item]
print(len(num_neutral2))

num_love = [item for item in dicList if 'love' in item]
print(len(num_love))

num_gratitude = [item for item in dicList if 'gratitude' in item]
print(len(num_gratitude))

num_disapproval = [item for item in dicList if 'disapproval' in item]
print(len(num_disapproval))

num_amusement = [item for item in dicList if 'amusement' in item]
print(len(num_amusement))

num_disappointment = [item for item in dicList if 'disappointment' in item]
print(len(num_disappointment))

num_realization = [item for item in dicList if 'realization' in item]
print(len(num_realization))

num_admiration = [item for item in dicList if 'admiration' in item]
print(len(num_admiration))

num_annoyance = [item for item in dicList if 'annoyance' in item]
print(len(num_annoyance))

num_confusion = [item for item in dicList if 'confusion' in item]
print(len(num_confusion))

num_optimism = [item for item in dicList if 'optimism' in item]
print(len(num_optimism))

num_excitement = [item for item in dicList if 'excitement' in item]
print(len(num_excitement))

num_caring = [item for item in dicList if 'caring' in item]
print(len(num_caring))

num_nervousness = [item for item in dicList if 'nervousness' in item]
print(len(num_nervousness))

num_desire = [item for item in dicList if 'desire' in item]
print(len(num_desire))

num_remorse = [item for item in dicList if 'remorse' in item]
print(len(num_remorse))

num_approval = [item for item in dicList if 'approval' in item]
print(len(num_approval))

num_joy = [item for item in dicList if 'joy' in item]
print(len(num_joy))

num_embarrassment = [item for item in dicList if 'embarrassment' in item]
print(len(num_embarrassment))

num_surprise = [item for item in dicList if 'surprise' in item]
print(len(num_surprise))

num_curiosity = [item for item in dicList if 'curiosity' in item]
print(len(num_curiosity))

num_anger = [item for item in dicList if 'anger' in item]
print(len(num_anger))

num_grief = [item for item in dicList if 'grief' in item]
print(len(num_grief))

num_disgust = [item for item in dicList if 'disgust' in item]
print(len(num_disgust))

num_pride = [item for item in dicList if 'pride' in item]
print(len(num_pride))

num_relief = [item for item in dicList if 'relief' in item]
print(len(num_relief))

num_fear = [item for item in dicList if 'fear' in item]
print(len(num_fear))

emotion = np.array([len(num_admiration), len(num_amusement), len(num_approval), len(num_caring),len(num_desire),len(num_excitement),len(num_gratitude),len(num_joy),len(num_love),len(num_optimism),len(num_pride),len(num_relief),len(num_anger),len(num_annoyance),len(num_disappointment),len(num_disapproval),len(num_disgust),len(num_embarrassment),len(num_fear),len(num_grief),len(num_nervousness),len(num_remorse),len(num_sadness),len(num_confusion),len(num_curiosity),len(num_realization),len(num_surprise), len(num_neutral2)])
labels2 = np.array(['admiration', 'amusement', 'approval', 'caring', 'desire', 'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief', 'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness', 'confusion', 'curiosity', 'realization', 'surprise', 'neutral'])


f = plt.figure()
plt.pie(emotion, labels=labels2)
plt.show()
f.savefig('emotion.pdf')
