# Nucleus Segmentation and Analysis in Breast Cancer with the MIScnn Framework

[![arXiv](https://img.shields.io/badge/arXiv-2206.08182-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2206.08182)

This repository provides the source code for the paper: "Nucleus Segmentation and Analysis in Breast Cancer with the MIScnn Framework" on arXiv.

## Abstract
The NuCLS dataset contains over 220.000 annotations of cell nuclei in breast cancers. We show how to use these data to create a multi-rater model with the MIScnn Framework to automate the analysis of cell nuclei. For the model creation, we use the widespread U-Net approach embedded in a pipeline. This pipeline provides besides the high performance convolution neural network, several preprocessor techniques and a extended data exploration. The final model is tested in the evaluation phase using a wide variety of metrics with a subsequent visualization. Finally, the results are compared and interpreted with the results of the NuCLS study. As an outlook, indications are given which are important for the future development of models in the context of cell nuclei.