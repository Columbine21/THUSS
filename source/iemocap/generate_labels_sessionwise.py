import os
import numpy as np
import json
import random
from pathlib import Path

Path('source/iemocap/labels_sess').mkdir(exist_ok=True)
with open('source/iemocap/metalabel.json', 'r') as f:
    metalabel = json.load(f)

labeldict = {
    'ang': 'anger',
    'hap': 'happy',
    'exc': 'happy',
    'sad': 'sad',
    'neu': 'neutral'
}

def in_session(speakerset, audioname):
    audio_gender = audioname[-8]
    audio_session = audioname[4]
    for speaker in speakerset:
        gender = speaker[0]
        session = speaker[1]
        if gender == audio_gender and session == audio_session:
            return True
    return False

all_speakers = []

for i in range(5):
    total_trn_count, total_tst_count = 0, 0
    sess = i + 1
    test_speakers = [f'M{sess}', f'F{sess}']
    labels = {
        'Train': {},
        'Val': {},
        'Test': {}
    }
    for audio in os.listdir('source/iemocap/Audio_16k'):
        label_key = metalabel[audio]
        if label_key not in labeldict:
            continue
        label = labeldict[label_key]
        if in_session(test_speakers, audio):
            labels['Test'][audio] = label
            total_tst_count += 1
        else:
            labels['Train'][audio] = label
            total_trn_count += 1
    print(f"{i}: Train Ins: {total_trn_count}, Test Ins: {total_tst_count}.")
    with open(f'source/iemocap/labels_sess/label_{i+1}.json', 'w') as f:
        json.dump(labels, f, indent=4)
