# XLS-R-ELAN 0.1.0

XLS-R-ELAN integrates the automatic speech recognition (ASR) methods offered by
[XLS-R](https://huggingface.co/facebook/wav2vec2-xls-r-300m) ([Babu et al.
2021](https://arxiv.org/abs/2111.09296)) into 
[ELAN](https://tla.mpi.nl/tools/tla-tools/elan/), allowing users to apply
fine-tuned XLS-R ASR models to multimedia sources linked to ELAN transcripts
from directly within ELAN's user interface.

This repository also contains several scripts used to fine-tune XLS-R models
using [Digital Research Alliance of Canada](https://alliancecan.ca)
high-performance computing facilities.  While specific to one specific language
(Tsuut'ina), both the fine-tuning scripts (Python) and SLURM job definitions
(sh) found in the `finetune-srs` directory may be useful as examples of how to
fine-tune XLS-R models, in general.

## Requirements and installation

XLS-R-ELAN makes use of several of other open-source applications and
utilities:

* [ELAN](https://tla.mpi.nl/tools/tla-tools/elan/) (tested with ELAN 6.4
  under macOS 13.1)
* [Python 3](https://www.python.org/) (tested with Python 3.10)
* [ffmpeg](https://ffmpeg.org)

XLS-R-ELAN is written in Python 3, and also depends on the following
Python packages, all of which should be installed in a virtual environment:

* [Hugging Face Transformers](https://pypi.org/project/transformers/) (tested
  with v4.11.3) and
  [Hugging Face Hub](https://pypi.org/project/huggingface-hub/) (tested with
  v0.1.0), installed with all of their dependencies in the same virtual
  environment as XLS-R-ELAN.
* [NumPy](https://numpy.org/) (tested with v1.21.5)
* [PyTorch](https://pypi.org/project/torch/) (tested with v1.11.0)
* [pydub](https://github.com/jiaaro/pydub), installed in the same
   virtual environment as XLS-R-ELAN (tested with v0.25.1)
* [CTCWordBeamSearch](https://github.com/githubharald/CTCWordBeamSearch),
   installed in the same virtual environment as the above.

Under macOS 13.1, the following commands can be used to fetch and install the
necessary Python packages:
```
git clone https://github.com/coxchristopher/xls-r-elan
cd xls-r-elan

python3 -m virtualenv venv-xls-r-elan
source venv-xls-r-elan/bin/activate

pip install transformers==4.11.3
pip install huggingface_hub==0.1
pip install numpy
pip install torch
pip install pydub

git clone https://github.com/githubharald/CTCWordBeamSearch
cd CTCWordBeamSearch
pip install .
```

In earlier testing under macOS 12.2.1, it was necessary to edit `setup.py`
in the `CTCWordBeamSearch` package before compiling and installing the package
with `pip install .`, adding `extra_compile_args` to the Extension definition:

```
word_beam_search_ext = Extension('word_beam_search', sources=src, \
    include_dirs=inc, language='c++', extra_compile_args=["-std=c++11"])
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```
  
Once all of these tools and packages have been installed, XLS-R-ELAN can
be made available to ELAN as follows:

1. Edit the file `xls-r-elan.sh` to specify (a) the directory in
   which ffmpeg is located, and (b) a Unicode-friendly language and
   locale (if `en_US.UTF-8` isn't available on your computer).
2. To make XLS-R-ELAN available to ELAN, move your XLS-R-ELAN directory
   into ELAN's `extensions` directory.  This directory is found in different
   places under different operating systems:
   
   * Under macOS, right-click on `ELAN_6.4` in your `/Applications`
     folder and select "Show Package Contents", then copy your `XLS-R-ELAN`
     folder into `ELAN_6.4.app/Contents/app/extensions`.
   * Under Linux, copy your `XLS-R-ELAN` folder into `ELAN_6-4/app/extensions`.
   * Under Windows, copy your `XLS-R-ELAN` folder into `C:\Users\AppData\Local\ELAN_6-4\app\extensions`.

Once ELAN is restarted, it will now include 'XLS-R speech recognition'
in the list of Recognizers found under the 'Recognizer' tab in Annotation Mode.
The user interface for this recognizer allows users to enter the settings needed
to apply automatic speech recognition to annotations on a selected tier, based
on the contents of a selected audio recording that has been linked to this ELAN
transcript.  Additional settings (e.g., whether or not to apply a word beam
search) can be configured through the recognizer interface, as well.

Once these settings have been entered in XLS-R-ELAN, pressing the `Start`
button will begin applying XLS-R's automatic speech recognition to the selected
audio recording for each of the annotations on the selected tier.  Once that
process is complete, if no errors occurred, ELAN will allow the user to load
the resulting tier with the automatically recognized segments and text into
the current transcript.

## Limitations

This is an alpha release of XLS-R-ELAN, and has only been tested under macOS
(13.1) with Python 3.10.  No support for Windows or Linux is included in this
version.

In several places, the current XLS-R-ELAN source code includes references to
the absolute paths of specific fine-tuned XLS-R models used in local testing.
For now, these will need to be changed manually in `xls-r-elan.py` when using
XLS-R-ELAN with any other fine-tuned XLS-R models.

## Acknowledgements

Thanks are due to the developers of [XLS-R](https://arxiv.org/abs/2111.09296),
as well as
[Hugging Face](https://huggingface.co/docs/transformers/model_doc/xls_r)
for making pre-trained models available.  
[Cécile Macaire](https://hal.science/hal-03429051)'s Jupyter notebook on
fine-tuning an XLS-R model for Yongning Na (Sino-Tibetan; ISO 639-3: nru)
and the Python scripts that accompanied her 2021
[internship with Lacito](https://github.com/macairececile/internship_lacito_2021) 
provided helpful information on working with XLS-R models in the context of
work with low-resource languages.  Thanks, as well, to
[Han Sloetjes](https://www.mpi.nl/people/sloetjes-han)
for his help with issues related to ELAN's local recognizer specifications.

## Citing XLS-R-ELAN

If referring to this code in a publication, please consider using the following
citation:

> Cox, Christopher. 2023. XLS-R-ELAN: An implementation of XLS-R automatic speech recognition as a recognizer for ELAN. Version 0.1.0.

```
@manual{cox23xlsrelan,
    title = {XLS-R-ELAN: An implementation of XLS-R automatic speech recognition as a recognizer for ELAN},
    author = {Christopher Cox},
    year = {2023}
    note = {Version 0.1.0},
    }
```
