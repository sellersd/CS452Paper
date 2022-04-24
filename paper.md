# Critical and Interpretive Analysis of Classical Texts Using Natural Language Processing Methods

# Abstract
---

Objectivity of interpretive text analysis has been a topic of contention throughout the history of the social sciences. The source of contention is an inherent suspicion of how sources of bias could influence the interpretive process.   The development of unsupervised machine-learning tools for text analysis have the promise of providing evidence for interpretations in a fashion that is both scalable and less subject to the biases from the preconceptions of interpreters.

In this work, we consider a set well-understood literary sources that have evolved from Early Modern English to Modern English to test a series of hypothesizes about how unsupervised machine learning goes about representing patterns of thoughts in text and how such tools may be used to aid interpreters in the analysis of texts, classical and modern.  At this phase of the project, we strive to identify methods that can compare two semantic models and analyze the differences to find indicators of bias.

# Introduction
---
In the social sciences, translation of primary sources can often require the translation from the original language to the language of the researchers. Historically, this has been accomplished by researchers translating texts between languages based upon knowledge of the original language. This has presented difficulties in cases such as deciphering Egyptian hieroglyphs where knowledge of the Egyptian language had been lost. This code was eventually cracked by utilizing the Rosetta stone, a parallel text using both Ancient Greek and Demotic script and hieroglyphics from Egypt. This methodology is still used, among others, but has not surpassed the inherent error that arises from both implicit and explicit biases. (Pryzant, et al., 2020) Removing subjectivity from a translation can lead to a more accurate and reliable translation. Using a parallel corpus, such as the Rosetta stone, can aide in creating these translations, but is still subject to the interpretations of the authors.

“Parallel corpora are a valuable resource for linguistic research and natural language processing (NLP) applications.” (Christodouloupoulos & Steedman, 2014) One of the most widely translated books in the world is the Bible. From texts written in ancient languages such as Aramaic, Hebrew, and Latin to modern languages including English, Mandarin Chinese, and many others, the Bible provides a significant number of parallel corpora for use in NLP. While the bible consists of roughly 800k words as compared to some other corpora containing 60M words, there are still advantages to utilizing this resource. Namely, the wealth of translations as well as the specific naming and numbering conventions used in the Bible. The latter allows for specific verse by verse comparisons. (Christodouloupoulos & Steedman, 2014)

By utilizing known parallel corpora, specifically ,the Wycliffe translation of the Bible dating from the late fourteenth century, and the Open English Bible the authors will train a neural network to conduct machine translation of Late Middle English Texts to modern English. During the training, the authors will measure the accuracy of translation by investigating common metrics such as NIST, BLEU, WER, and PER. (Popović & Ney, 2011; Turian, et al., 2006) Upon reaching a satisfactory level of accuracy, other Late Middle English documents will be translated and a metric will be used to compare human translated versions with these machine translated texts to look for biases with the investigation starting with the work of Recasens et al. (2013).

# Background - David
---
## Translation
	1. Neural Network
	2. Metrics
	3. Translation
	4. Machine Translation
	5. Supervised and Unsupervised Learning
	6. Parallel Corpora vs. non-Parallel Corpora

Spreading knowledge and work through time and space often requires translating works between languages. Whether these are works of antiquity, such as Homer's works, works from the Renaissance, such as Newton's Principia Mathematica, or contemporary works, in order to share these with people around the world necessitates the translation from their original language to the language of the target audience. While this is non-trivial for modern works that need to be translated into another common modern language, it is feasible to find a translator to transcribe the work into the new language. However, this process is much more difficult, and in some cases impossible, due to the lack of people fluent in some more obscure languages translations can be difficult, and in the case of dead languages can be impossible. In addition to the previous difficulties, human translated works are subject to errors and implicit and explicit biases. This leads to the need for a more reliable manner of translating documents in a manner that is objectively and measurably correct.

This leads to the need for a new manner of translation, namely Machine Translation. The algorithms of interest in this project are based upon neural networks.

![image][Images/s2s_encoder.png "Sequence to Sequence Encoder"]
<img src="../Images/s2s_encoder.png"/>

Sequence-to-sequence encoder

The sequences involved in the sequence-to-sequence encoder are from parallel corpora, which are two bodies of work, one written in the source language the other written in the target language. The RNN attempts to translate from the target language to the source language.

Machine learning can utilize either supervised or unsupervised learning. With supervised learning, the model is trained on labeled data so that each piece of data has a corresponding label identifying what the model should generate for that piece of data. While this would be a relatively simple manner to generate a translation using a dictionary, it fails to take into account the nuances of the languages and the fact that the grammar rules differ between languages.

With unsupervised learning, the model is trained without clearly labeled data and attempts to learn to identify data by detecting patterns in the training data.

Generating a machine translation of a work can result in a translated work that is free from the inherent inaccuracies present in human translated works, but it cannot be assumed that translation is suitably accurate.

https://commons.wikimedia.org/wiki/File:Neural_Network_Dropout.svg

# Research Background - David
---
Traditionally, works were translated between languages by human translators. As computing technology evolved, interest began to develop in utilizing machines to perform these translations. In the 1960's, the attempts to translate using the computers of the day were based off of complex rule sets developed by engineers to follow the manual translation.  Due to the variability in the use of natural languages this approach failed to be effective and a new approach was needed. With the increased processing power of computers decreased, a move was made to automate the development of the rule set for translating natural language texts.  (Deep Learning Book) Developing this ruleset is key to developing the machine translation (MT) model. In order to develop the ruleset training data must be used to relate the source and target languages. According to Christodouloupoulos & Steedman, the Bible is a particularly useful corpus due to its wide availability and reliable translations into many languages. (2014) These translations create parallel corpora that allow the model to be trained on human translated works in order to emulate these known good translations.

# The Capstone Project - Split
	1. Description of team
		a. Learning curve
	2. Description of project
		a. scope
		b. Training data
			I. Consisted of the four gospels
			II. 3779 paired lines
			III. Wycliffe 83139 words
			IV. OEV 84264 words
	3. Agile methodology

# Implementation - John
---
	1. Three algorithms
		a. Versioning between Python and TensorFlow
		b. Failed to translate meaningfully
		c. Semi-successful algorithm
			I. Corpora size

Explored multiple translation models and attempted to adapt them to Late Middle English to Modern English translations.
The first explored model was a French to English model that the authors could not make function due to versioning errors between Python and the Tensorflow library.
The second model was a German to English model that would function, but would not output meaningful translations. The resulting translations were like such "and the the the the the the the etc"
The third model was a Spanish to English model obtained from the Keras-io repository under the Apache license. This model was succesfully adapted to a Late Middle English to Modern English sequence to sequence transformer. Changes to the input size were needed as the goal was to input whole Bible verses as a sequnce and the existing model was tailored for short 2-4 word sequences,  whereas the Bible's longest verse is 80 words long. The BLEU score metric was added to the resultant translation from the model, but would not output meanignful scores.'


# Future Work
---
Though the Bible is a very convenient parallel data set due to having numbered and aligned verses and being widely translated,  it is quite small in size for a parallel corpus. The Bible is approximately 800 thousand words in size whereas a common modern parallel corpus such as the Europarl corpus has an average of 60 million words per language.  (Christodouloupoulos & Steedman, 2014)

The BLEU score of the implemented machine translation was derived. Future efforts could gain more insight into the quality of the translation by using other metrics such as NIST, WER, and PER.

The overall aim of the project was to compare our Bible trained model's translation of other Middle English works to their humanly translated versions and scan for bias in the human translation in comparison to the machine translation. A sufficiently trained model was not developed in time for work on bias detection to proceed. Future researchers can begin their work with the model developed in this work trained on larger corpora and examine its translations for bias.

The computational time per epoch was approximately 25 Seconds on an AMD 2700x processor for a 80k word sample of the Bible.  Future efforts with larger corpora may need to set up GPU utilization options from the TensorFlow library to keep computational time per epoch manageable.

# Summary
---
