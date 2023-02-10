#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Apply a fine-tuned XLS-R model to the annotations on a given tier in an ELAN
# transcript, returning as output a new tier with the model's predictions for
# each of the provided annotations (as part of a local recognizer in ELAN).
#
# If a corpus (i.e., a text file containing a bunch of Tsuut'ina text) is
# provided, this recognizer also applies a word beam search (Scheidl 2018),
# which uses both a dictionary and language model derived from the provided
# corpus to refine the XLS-R model's predictions.
#
# See:
#
#   https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
#   https://github.com/macairececile/internship_lacito_2021/blob/main/notebooks/Fine-Tune%20XLSR-Wav2Vec2%20-%20Na.ipynb
#   https://github.com/githubharald/CTCWordBeamSearch
#   https://towardsdatascience.com/word-beam-search-a-ctc-decoding-algorithm-b051d28f3d2e

# TODO: 
#
#   * Add some pre-processing to the corpus texts that get submitted! (all
#     kinds of bizarre Unicode characters can get swapped in, multiple spaces,
#     etc., all of which should get cleaned up first)
#
#   * Retrain the model to (a) not worry about word-initial glottal stops any
#     more, and (b) to do hyphens, commas, and periods (and if it turns out 
#     to be terrible at punctuation, handle as much of it in post-hoc adjust-
#     ments as possible
#
# FROMHERE

# Installation notes:
#
# !pip install transformers==4.11.3
# !pip install huggingface_hub==0.1
# !pip install numpy
# !pip install torch
# !pip install pydub
#
# Also have to download and install word beam search directly from GitHub (at
# https://github.com/githubharald/CTCWordBeamSearch): 
#
#   git clone https://github.com/githubharald/CTCWordBeamSearch
#   cd CTCWordBeamSearch
#   pip install .
#
# This compiles a C++ Python extension, which is fine, but didn't work right
# out of the box (on macOS 12.2.1, Intel).  It ended up being necessary to
# edit 'setup.py', adding "extra_compile_args" to the Extension definition:
#
#   word_beam_search_ext = Extension('word_beam_search', sources=src, \
#       include_dirs=inc, language='c++', extra_compile_args=["-std=c++11"])
#                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import atexit
import codecs
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pydub
import torch
import transformers
import word_beam_search

import retone_srs

asr_model_dir = "/Users/chris/Desktop/TLL/Code/xls-r-elan/wav2vec2-xls-r-300-srs-test1/checkpoint-30400"
tone_model_dir = "/Users/chris/Desktop/TLL/Code/xls-r-elan/wav2vec2-xls-r-300-srs-tones/checkpoint-30400"

model_dir = asr_model_dir

tmp_audio = tempfile.NamedTemporaryFile(suffix = '.wav', delete = False)
tmp_audio.close()

@atexit.register
def cleanup():
    # When this recognizer ends (whether by finishing successfully or when
    # cancelled), remove all of the temporary files that this script created
    # that 'tempfile' won't do itself.
    if tmp_audio:
        os.remove(tmp_audio.name)

# Begin by tracking down the ffmpeg(1) executable that this recognizer will use
# to process audio materials.  If ffmpeg(1) doesn't exist in the current path, 
# exit now to save everyone some heartbreak later on.
ffmpeg = shutil.which('ffmpeg')
if not ffmpeg:
    sys.exit(-1)

# Read in all of the parameters that ELAN passes to this local recognizer on
# standard input.
params = {}
for line in sys.stdin:
    match = re.search(r'<param name="(.*?)".*?>(.*?)</param>', line)
    if match:
        params[match.group(1)] = match.group(2).strip()

# If 'output_tier' wasn't specified, bail out now (and skip all of the
# processing below, since ELAN won't be able to read the results without an
# output tier being specified, anyway).
if not params.get('output_tier'):
    print("ERROR: no output tier specified!", flush = True)
    sys.exit(-1)

# With those parameters in hand, grab the 'input_tier' parameter, open that
# XML document, and read in all of the annotation start times, end times,
# and values.
##print("PROGRESS: 0.1 Loading annotations on input tier")
annotations = []
with open(params['input_tier'], 'r', encoding = 'utf-8') as input_tier:
    for line in input_tier:
        match = re.search(r'<span start="(.*?)" end="(.*?)"><v>(.*?)</v>', line)
        if match:
            annotation = { \
                'start': int(float(match.group(1)) * 1000.0), \
                'end' : int(float(match.group(2)) * 1000.0), \
                'value' : match.group(3) }
            annotations.append(annotation)

# Use the model that corresponds to the function that the user selected in the
# recognizer ("Transcribe" or "Retone").
if params['mode'] == 'Transcribe':
    model_dir = asr_model_dir
else:
    # When retoning text, we can skip any annotations that don't contain any
    # text other than whitespace.
    annotations = [a for a in annotations if a['value'].strip()]
    model_dir = tone_model_dir

# Use ffmpeg(1) to convert the 'source' audio file into a temporary 16-bit
# mono 16KHz WAV, then load that into pydub for quicker processing.
##print("PROGRESS: 0.2 Converting source audio", flush = True)
subprocess.call([ffmpeg, '-y', '-v', '0', \
    '-i', params['source'], \
    '-ac', '1',
    '-ar', '16000',
    '-sample_fmt', 's16',
    '-acodec', 'pcm_s16le', \
    tmp_audio.name])

converted_audio = pydub.AudioSegment.from_file(tmp_audio.name, format = 'wav')

# Now that we have our audio ready to go, load the model and the associated
# processor.
model = transformers.Wav2Vec2ForCTC.from_pretrained(model_dir)
processor = transformers.Wav2Vec2Processor.from_pretrained(model_dir)

# If we were given a corpus to use as part of a word beam search to refine the
# XLS-R model's predictions (and we're in transcription mode), load it now.
use_word_beam_search = ('corpus' in params) and \
    os.path.isfile(params['corpus']) and (params['mode'] == 'Transcribe')
if use_word_beam_search:
    corpus = codecs.open(params['corpus'], 'r', 'utf-8').readlines()[0].strip()

    # TODO: lowercase and clean up the text in the corpus before doing
    #       anything else! (FIXME FROMHERE)

    # Get all of the characters in the model's vocabulary in a single list,
    # leaving out the special characters added as part of modelling, leaving
    # out the final "[PAD]" character (which would normally be the last item
    # in this list).
    vocab = list(processor.tokenizer.get_vocab().keys())[:\
        processor.tokenizer.vocab_size - 1]
    vocab[vocab.index('|')] = ' '
    vocab[vocab.index('[UNK]')] = '?'
    vocab_chars = ''.join(vocab)

    # Get all of the word-forming characters into a single string.
    word_chars = ''.join([c for c in vocab if not c in [' ', '?']])

    # Initialize a word beam search, using the given corpus as the source for
    # an internal dictionary and (unigram and bigram) language model and the
    # given lists of characters to determine what can make up an orthographic
    # word in Tsuut'ina.
    wbs = word_beam_search.WordBeamSearch(25, 'NGrams', 0.0, \
        corpus.encode('utf-8'), vocab_chars.encode('utf-8'), \
        word_chars.encode('utf-8'))

# Process each of the annotations that we were given, using the above model
# and processor to generate predictions that are stored as output annotations
# in 'output'.
num_annotations = len(annotations)
for (i, a) in enumerate(annotations):
    # Extract the audio for this annotation from the converted audio that was
    # loaded using pydub, then convert that into a NumPy array with dtype =
    # np.float32 and a value range normalized to [-1.0, 1.0].  The line that
    # converts pydub samples to a NumPy float32 array comes from:
    #
    #   https://github.com/jiaaro/pydub/blob/master/API.markdown
    clip = converted_audio[a['start']:a['end']]
    samples = clip.get_array_of_samples()
    speech = np.array(samples).T.astype(np.float32) / \
        np.iinfo(samples.typecode).max

    # Process this snippet, producing input that the model can work with,
    # then use the model to derive a logits vector that gives the (log-)
    # likelihood of each element in our vocabulary at each point.  Taking the
    # argmax of those logits, then decoding them back into the actual
    # characters that each element in the vocabulary represents, gives us our
    # predicted text.
    input_dict = processor(speech, return_tensors = "pt", padding = True, \
        sampling_rate = 16000)
    logits = model(input_dict.input_values).logits

    label = ''
    if use_word_beam_search:
        # 'word_beam_search.compute' expects that softmax has already been
        # applied, and that the array have the shape (T, B, C + 1), where
        # T is the number of time steps, B is the batch size, and C is the
        # number of characters.  The logits returned by the fine-tuned model
        # above have the shape (B, T, C + 1), and the scores for each
        # vocabulary token haven't run through softmax, yet, so we apply
        # softmax and swap those two axes here first.
        wbs_input = np.swapaxes(logits.softmax(dim = -1).detach().numpy(), 0, 1)
        label_ids = wbs.compute(wbs_input)[0]
        label = ''.join([vocab_chars[label_id] for label_id in label_ids])
    else:
        pred_ids = torch.argmax(logits, dim = -1)[0]
        label = processor.decode(pred_ids)

    # If we have been asked to retone the text on the given tier, use the
    # tones that the model has predicted (in 'label'), then do so now.
    if params['mode'] == 'Retone':
        retoned_text = retone_srs.retone(a['value'], label)
        if retoned_text.startswith("*"):
            a['output'] = '*' + a['value']
        else:
            a['output'] = retoned_text
    else:
        a['output'] = label

    print("PROGRESS: %.2f Processing annotation %d of %d" % \
        (i / num_annotations, i + 1, num_annotations), flush = True)

# Write all of the output annotations to 'output_tier'.
with open(params['output_tier'], 'w', encoding = 'utf-8') as output_tier:
    # Write document header.
    output_tier.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    output_tier.write('<TIER xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="file:avatech-tier.xsd" columns="XLS-R-ELAN-Output">\n')

    # Write out annotations and recognized text (e.g., '<span start="17.492"
    # end="18.492"><v>OUTPUT</v></span>').
    for a in annotations:
        output_tier.write(\
            '    <span start="%s" end="%s"><v>%s</v></span>\n' %\
            (a['start'], a['end'], a['output']))

    output_tier.write('</TIER>\n')

# Finally, tell ELAN that we're done.
print('RESULT: DONE.')
