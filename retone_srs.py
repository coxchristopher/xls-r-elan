#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# s = unicodedata.normalize('NFKC', s) # canonical, precomposed form

import re
import unicodedata

_toned_vowels =    'áāaàíīiìóōoòúūuùÁĀAÀÍĪIÌÓŌOÒÚŪUÙ'
_toneless_vowels = 'aaaaiiiioooouuuuAAAAIIIIOOOOUUUU'
_detone_vowels = ''.maketrans(_toned_vowels, _toneless_vowels)

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

def retone(text, tones):
    # Remove all tone marking from the source text and turn the target tones
    # into a list (e..g, ['LL', 'M', 'HH'])
    text = text.translate(_detone_vowels)
    tones = tones.split(" ")

    # If the number of vowels in the original text doesn't match the number
    # of vowels recognized by the auto-toner, then return the original text
    # without any changes, except for a prepended "*".
    vowels = re.findall('([%s]+)' % _toned_vowels, text)
    if len(vowels) != len(tones):
        return "*" + text
   
    def rewrite_tone(v):
        t = tones.pop(0)                                # e.g., 'HH'
        v = v * len(t)                                  # e.g., 'a' --> 'aa'
        return ''.join(_tone_and_vowel_to_toned_vowel[tv] for tv in zip(t, v))

    # For each vowel (short or long) in the original text, hand over its
    # first character to the inner function 'rewrite_tone' to adjust the
    # vowel length and tone to match the corresponding tone in 'tones'
    # (e.g., 'a' in first match, 'HH' in tones[0] --> 'áá' as output).
    return re.sub('([%s]+)' % _toned_vowels, lambda m: \
        rewrite_tone(m.group(0)[0]), text)

# Turn a string of orthographic Tsuut'ina text into a space-separated string
# of of "MM L HH"-style tone labels (for converting sentences in the original
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

    return ' '.join(tones)


# test = "dosak'a"
# tones = "MM MH ML"
# 
# print(retone(test, tones))
