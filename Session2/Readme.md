 # Session 2: 
This session introduced to MFA functions like corpus validation, training of acoustic models, and training of g2p models and their utilizatiion in the creation of dictionaries.

## Tutorials

### 0 Useful functions in PRAAT and Audacity
This tutorial introduced some options for pre-segmenting the utterances in a corpus using either PRAAT or Audacity. After rough segmentations are done, the scripts in the scripts folder extract all the utterances and create a corpus folder that is conform with the corpus structure needed for the MFA.

### 1 Corpus validation

This tutorial shows how to use the validate function of the MFA in order to detect errors in a corpus and a dictionary.

### 2 G2P model: dictionary creationi

Grapheme-to-phoneme (g2p) models are introduced in this tutorial and how theiy can be used to modify or create pronunciation dictionaries.

### 3 Training of acoustic models

This tutorial shows how to train a new acoustic model that is necessary for the forced alignment. A dataset on Basque is used for illustratioin.

### UseCase 1: French audbook alignment 2

This section show how to use Audacity/Praat for presegmentation and the subsequent alignment of the data.

### UseCase 2: Training of a new acoustic model on Mandarin speech data

This tutorial guides through the training of a new acoustic model using a large dataset (~10 h) of Mandarin

### UseCase 3: Xhosa data

In this notebook we work on Xhosa data, a language that has no acoustic model, dictionary and g2p model in the MFA repository.