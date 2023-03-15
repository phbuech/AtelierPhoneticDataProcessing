# Session 1: Introduction to the Montreal Forced Aligner
This session introduces the Montreal Forced Aligner (MFA) with some basic examples. If it was not possible to install the MFA, consult MFA_installation.pdf. If you were able to install the MFA but it worked not properly, follow the instructions too in order to remove the environment and install an older version of the MFA in a new environment (this has to be done, because it seems that there are some issues in the recent update).

#### For Windows users:
If you installed Anaconda, but it was not possible to install a Python package (e.g., by the command: conda install -c conda-forge scipy) due to an OPENSSL error, moving 2 files should solve this issue.
Go to
```
C:/Users/your_user_name/anaconda3/Library/bin/
```
and copy the two files
- libcrypto-1_1-x64.dll
- libssl-1_1-x64.dll
into the folder:
```
C:/Users/your_user_name/anaconda3/DLLs/
```
If you follow the instructions in the MFA_installation.pdf, the installation via
```
conda install -c conda-forge montreal-forced-aligner=2.2.0
```
should be successfull.

# Tutorials
The following is a list of different use cases, with applications on data from different languages and genres (speech synthesis corpus, audiobook, broadcast speech). You received the link for the download of the datasets in a separate email. Each folder contains not only the raw data needed for the alignment procedure, but also the results of

### Use Case 1a: French alignments
This tutorial demonstrates the basic functionality of the MFA using a French corpus as an example. The data is from the [SIWIS French Speech Synthesis dataset](https://datashare.ed.ac.uk/handle/10283/2353). You will find .lab files containing the transcriptions in the download folder.
### Use Case 1b: Modifying the dictionary
You may see that the "vanilla" dictionary does not account for variability in pronunciation. This tutorial demonstrates how to use and modify the dictionary in order to account for certain variations.
### Use Case 2: Aligning a French audiobook
This tutorial demonstrates how to align utterances from an audiobook. The data is from a freely available audiobook "Le dernier jour d'un condamné" of Victor Hugo from [LibriVox](https://librivox.org/le-dernier-jour-by-victor-hugo/). The corresponding text can be found [here](https://fr.wikisource.org/wiki/Le_Dernier_Jour_d%E2%80%99un_Condamn%C3%A9/01). Note that the data in the download folder were already converted into wav files.
### Use Case 3: English alignments
English data from the [DARPA TIMIT](https://www.kaggle.com/datasets/mfekadu/darpa-timit-acousticphonetic-continuous-speech) dataset were aligned in this tutorial. .lab-files containing the corresponding transcriptions of the utterances are provided in the download folder. Two different acoustic models and dictionaries were used for the alignments.
### Use Case 4: Aligning Swahili broadcast speech
This use case demonstrate how to align Swahili broadcast speech data. The full corpus is availble [here](http://openslr.org/25/).

### Use Case 5: Aligning Hausa speech data
The last tutorial shows the alignment of Hausa data from the [Mozilla Common Voice](https://commonvoice.mozilla.org/) initiative. The entire dataset can be found [here](https://www.africanvoices.tech/language/hau). Two pretrained acoustic models are available; one is the hausa_cv model without accompanying dictionary, and one is the hausa_mfa model including a dictionary.
