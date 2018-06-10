# Introduction

- [X] InsightFace inference example (production ready architecture)

- [X] Face recognition demo with insightface (visualization missing, add later)

- [ ] InsightFace training pipeline

# What does this do if I know nothing about face recognitions?

This is a server, wrapping up with a frozen model, accepting a photo of face, then output a vector of 512 dimension to
describe it.

It means:

- [X] You pbbly need another pipeline before this to detect a face bounding box first

- [X] Then you can run this project to describe this face

- [X] Later on, it's up to your purpose, if your purpose is face alignment/detection/distinguish, you need another classifier after this to do the job

An example of this is as following, borrowed from [openface](https://cmusatyalab.github.io/openface/):

![arch](https://raw.githubusercontent.com/cmusatyalab/openface/master/images/summary.jpg)


# Demo Facial Recognition App

Yes Yes Yes, I know your are lazy. So I made a demo app for you with following architecture:

![sys - page 1](https://user-images.githubusercontent.com/8921629/41207031-e7e640fa-6cc2-11e8-8e36-029fe8d41885.png)


# How to run it

* Install Depenencies: `pip install -r requirements.txt`

* Download pre-trained [frozen model](https://drive.google.com/open?id=1Iw2Ckz_BnHZUi78USlaFreZXylJj7hnP) and put it under [`pretrained` folder](https://github.com/AIInAi/tf-insightface/tree/master/pretrained)

* Run example: `python apps/example.py`

* You shall be able to see terminal output a 512 element array representing face feature embedded

* Run demo: `python apps/demo.py`

* You sahll be able to see it output the architecture described above logs

# References

* [Deng, Jiankang, Jia Guo, and Stefanos Zafeiriou. "Arcface: Additive angular margin loss for deep face recognition." arXiv preprint arXiv:1801.07698 (2018).](https://arxiv.org/abs/1801.07698)

* Official Implementation (mxnet): [deepinsight/insightface](https://github.com/deepinsight/insightface)

* Third Party Implementation (tensorflow): [auroua/InsightFace_TF](https://github.com/auroua/InsightFace_TF)