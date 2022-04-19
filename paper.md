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
	1. Neural Network
	2. Metrics
	3. Translation
	4. Machine Translation
	5. Supervised and Unsupervised Learning
	6. Parallel Corpora vs. non-Parallel Corpora

# Research Background - David
---
	1. Selected works review

# The Capstone Project - Split
	1. Description of team
		a. Learning curve
	2. Description of project
		a. scope
	3. Agile methodology

# Implementation - John
---
	1. Three algorithms
		a. Versioning between Python and TensorFlow
		b. Failed to translate meaningfully
		c. Semi-successful algorithm
			I. Corpora size
			
# Future Work - John
---
	1. Expand to larger corpora
	2. Test different metrics
	3. Bias analysis in human translations
	4. GPU utilization

# Summary
---
