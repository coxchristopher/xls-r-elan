#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import unicodedata

import torchaudio
import numpy as np

#srs_audio_path = '/home/cdcox/data/tsuutina_data-20200601/wav/'
srs_audio_path = 'tsuutina_data/wav/'

def srs_process_row(row):
    s = row['sentence']

    # Remove double hyphens from the front end of a continued word.
    # (e.g., "dàsyas-- tł'a- --tł'áł" --> "dàsyastł'a- --tł'áł",
    #  "dàsyas--, tł'a-" --> "dàsyastł'a-")
    s = re.sub('--,?($| )', '', s)    

    # Remove double hyphens from the end of a continued word, leaving it
    # separated from the preceding (partial) word if it was separated from
    # one by a space. (e.g., "dàsyastł'a --tł'áł" --> "dàsyastł'a tł'áł")
    s = re.sub('(^| )--', '\\1', s)

    # Remove any "stylistic" hyphens (e.g., "iyí ìsíla - ichi ásch'ánísk'òłí -
    # ..." --> "iyí ìsíla ichi ásch'ánísk'òłí ...")
    s = re.sub(' -($| )', '\\1', s)

    # TODO: Consider removing any "stylistic" commas that don't appear
    # immediately after a partial word (e.g., "ùwat'iyi itsiy-la, k'àt'íní
    # ii" --> "ùwat'iyi itsiy-la k'àt'íní ii"). We'll see how the model
    # does first with these still in place.

    # Retain any remaining hyphens that separate suffixes and enclitics
    # (at least for now; we'll see how the model does with this).

    # Change spaces between words into pipes (so that they can be treated as
    # full-fledged characters to be predicted by the model).
    s = s.replace(' ', '|')

    # Create a phoneme representation of this sentence.
    row['phonemes'] = to_phonemes(s)
    row['tones'] = to_tones(s)
    row['sentence'] = s
    return row

def to_phonemes(s, tone = 'separate'):
    # Turn cleaned orthographic text into a list of (orthographically
    # represented) phonemes. (e.g., "ístłí" -> ["í", "s", "tł", "í"]).
    #
    # If requested, this method can also return a version of the phoneme
    # string where tones are represented as separate phonemes in their own
    # right after the corresponding vowel (e.g., "ístłí" -> ["i", "H", "s",
    # "tł", "i", "H"])

    # For now, we're not distinguishing vowels with /w/ and /j/ off-glides
    # from vowels followed by the corresponding off-glide.  I don't think
    # that's quite right phonetically, but we'll see what the model is able
    # to do with it.  (The alternative is not pretty, since the orthography
    # doesn't give us enough information to distinguish stem vowels with
    # off-glides from stem vowels that just happen to be followed by <y>
    # (e.g., "it'iyi" vs. "itsiyí", where the second <i> in both words has
    # a different pronunciation)
    pattern = re.compile(r"\s*(ts'|tł'|ch'|kw')|(UH|gw|kw|dz|dl|ts|tł|ch|sh|zh|gh|t'|k')|(áá|áa|áà|aá|aa|aà|àà|àa|àá)|(íí|íi|íì|ií|ii|iì|ìì|ìi|ìí)|(óó|óo|óò|oó|oo|oò|òó|òo|òò)|(úú|úu|úù|uú|uu|uù|ùú|ùu|ùù)|(.)\s*")
    x = ' '.join([m.group(m.lastindex) for m in pattern.finditer(s)])

    # For both tone represented as separate symbols (tone == 'separate') and
    # tone stripped out of the labels (tone == 'none').
    if tone != 'integrated':
        # Another nasty hack to turn high-tone vowels like <á> into sequences
        # like <aH> (and similarly for mid and low-tone vowels).  A look-up
        # table would probably be much more transparent...
        x = re.sub(r'([aiou])', lambda m: \
            unicodedata.normalize('NFKD', m.group(m.lastindex) + 'M')\
                .encode('ASCII', 'ignore')\
                .decode('UTF-8'), x)
        x = re.sub(r'([áíóú])', lambda m: \
            unicodedata.normalize('NFKD', m.group(m.lastindex) + 'H')\
                .encode('ASCII', 'ignore')\
                .decode('UTF-8'), x)
        x = re.sub(r'([àìòù])', lambda m: \
            unicodedata.normalize('NFKD', m.group(m.lastindex) + 'L')\
                .encode('ASCII', 'ignore')\
                .decode('UTF-8'), x)

        # Reorder the tone marking in two-vowel sequences (e.g., 'aà' ->
        # 'aMaL' above, then 'aMaL' -> 'aaML' here)
        x = re.sub(r'([aiou])([HML])([aiou])([HML])', '\\1\\3\\2\\4', x)

        # Hackishly make sure that level tones on long vowels are marked
        # with a single symbol. (In theory, this could be treated as a variable
        # that could be changed between models -- does distinguishing tone
        # on short vs. long vowels make a difference to the system's overall
        # performance?)
        #
        # TODO
        x = x.replace('HH', 'H')
        x = x.replace('MM', 'M')
        x = x.replace('LL', 'L')

        # Finally, insert spaces between vowels and tone marking.
        x = re.sub(r'([aiou]+)([HML]+)', '\\1 \\2', x)

    # Strip out all tone marking if requested.
    if tone == 'none':
        x = re.sub(r' [HML]+', '', x)

    x = re.sub(r'\s+', ' ', x).strip()
    return x.split(' ')

_toned_vowels = 'áāaàíīiìóōoòúūuùÁĀAÀÍĪIÌÓŌOÒÚŪUÙ'
_tone_and_vowel_to_toned_vowel = {
    ('H', 'a'): 'á',
    ('M', 'a'): 'a',
    ('L', 'a'): 'à',
    ('H', 'i'): 'í',
    ('M', 'i'): 'i',
    ('L', 'i'): 'ì',
    ('H', 'o'): 'ó',
    ('M', 'o'): 'o',
    ('L', 'o'): 'ò',
    ('H', 'u'): 'ú',
    ('M', 'u'): 'u',
    ('L', 'u'): 'ù',
    ('H', 'A'): 'Á',
    ('M', 'A'): 'A',
    ('L', 'A'): 'À',
    ('H', 'I'): 'Í',
    ('M', 'I'): 'I',
    ('L', 'I'): 'Ì',
    ('H', 'O'): 'Ó',
    ('M', 'O'): 'O',
    ('L', 'O'): 'Ò',
    ('H', 'U'): 'Ú',
    ('M', 'U'): 'U',
    ('L', 'U'): 'Ù',
}

_toned_vowel_to_tone_and_vowel = dict((v, k) for k, v in \
    _tone_and_vowel_to_toned_vowel.items())

# Turn a string of orthographic Tsuut'ina text into a space-separated string
# of of "MM|L|HH"-style tone labels (for converting sentences in the original
# training data into tone sequences for the model to use as training data for
# tone recognition)
def to_tones(text):
    # This code has "turn me into an ugly Python list comprehension one-liner"
    # written all over it.  Fighting temptation...
    tones = []
    vowels = re.findall('([%s]+)' % _toned_vowels, text)
    for vowel in vowels:
        tone_seq = ''
        for v in vowel:
            tone_seq += _toned_vowel_to_tone_and_vowel[v][0]
        tones.append(tone_seq)

    return '|'.join(tones)

# Preprocess dataset audio.
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(srs_audio_path + batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch
