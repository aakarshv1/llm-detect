# ML@Berkeley NMEP Project - AI Generated Text Detection Model

![ML@Berkeley NMEP Project](https://ml.berkeley.edu/static/media/mlab-logo-horizontal-small.3d4a6012.png)

## Introduction
This project focuses on developing a machine learning model to distinguish between essays written by students and those generated by large language models (LLMs). Inspired by a [Kaggle competition](https://www.kaggle.com/competitions/llm-detect-ai-generated-text), this initiative addresses the challenges posed by LLMs in academic and professional contexts.

![Demo](https://github.com/aakarshv1/llm-detect/blob/main/Demo%20pic.png)

## Objectives
- **Development of a Detection Model:** Create a model to differentiate between student-written and LLM-generated essays.
- **Accuracy and Efficiency:** Focus on improving the Area Under the Curve (AUC) metric for accuracy.
- **Benchmarking and Improvement:** Replicate top Kaggle entries and state-of-the-art techniques, followed by iterative enhancements.

## Background
The urgency of this issue is highlighted by the Kaggle competition, backed by Vanderbilt University and The Learning Agency Lab, due to the increasing proficiency of LLMs in mimicking human writing.

## Methodology
- **Data Analysis:** Use the dataset from the Kaggle competition, including both student-written essays and LLM-generated texts.
- **Model Development:** Start with replicating top-performing models from the competition and academic research.
- **Iterative Improvement:** Enhance the model using various machine learning techniques.
- **Evaluation:** Consistently use the AUC metric for performance measurement.

## Timeline
- **Weeks 1-2:** Data preparation and initial model development.
- **Week 3:** Iterative improvements and optimization.
- **Week 4:** Final evaluation and adjustments.

## Expected Outcomes
- A high-accuracy AI detection model, as measured by AUC.
- Insights into differentiating AI-generated and human-written text.
- Contributions to the technology for detecting AI-generated content.

## Approaches
- Byte-Pair Encoding (BPE) Tokenization
- Compresses text input to allow for pre-training of classifier models
- Vectorizer
- Ensemble

## Other Approaches Considered/Potential Improvements
- [Ghostbuster: Detecting Text Ghostwritten by Large Language Models](https://arxiv.org/abs/2305.15047)
- Deep Learning based approach, basically compares results generated by 3 different LLM models
- This framework works great, but is not entirely efficient, especially for Kaggle Competitions
- Would be great if we can incorporate this into our apporach as well in the future

## Credits
- Learned a lot from the open notebook from [Prem ChotePaint](https://www.kaggle.com/batprem)
- [Hugging face bpe tokenizer](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt)
  
## Results (On-Going)
![Kaggle LeaderBoard](https://github.com/aakarshv1/llm-detect/blob/main/Kaggle%20Leaderboard.png)


