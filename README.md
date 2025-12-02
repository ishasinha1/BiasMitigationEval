# Evaluating LoRA-Based Post-Training Techniques for Bias Mitigation in LLMs

For CS329R: Race and Natural Language Processing (Autumn 2025-26)

This paper seeks to build on this work by analysing how different post-training methods improve the safety of language models without detracting from their general capabilities, specifically in the context of sociocultural biases like racial and gender bias. We fine-tune the same instruction-tuned model, **Qwen3-0.6B** using one safety dataset but, the different fine-tuning methods of **Supervised Fine-tuning (SFT)** and **Direct Preference Optimization (DPO)** to compare the efficacy of these methods for improving its safety while retaining the general capabilities. By using a uniform safety dataset that is not restricted to only the biases on which we evaluate these methods, we can also try to understand how well safety-tuning generalizes across different types of harm. 

## Training Notebook

We fine-tune the model using SFT and DPO through a LoRA adapter, on the Anthropic/hh-rlhf dataset. The notebook where data is prepared, and training takes place is: `CS329R_DPO_SFT.ipynb`.

The trained models can be found on HuggingFace:

DPO: [https://huggingface.co/ishasinha1/Qwen3-0.6B-DPO-Safety](url)

SFT: [https://huggingface.co/ishasinha1/Qwen3-0.6B-SFT-Safety
](url)

## General Capabilities Notebook

We evaluate the models on the MMLU dataset's test split (~14k general examples of multiple-choice questions) to confirm that fine-tuning on safety data has not degraded general model capabilities. The notebook for this is `General_Capabilities.ipynb`.  _Remember to modify the filepath!_

The results can be found in the final paper.

## Evaluation Notebook

The resulting models are evaluated against the base model in the notebook: `CS329R_Evaluation.ipynb`. We run inference to generate responses to test data, and then extract evaluation metrics from a pre-trained classifier and a reward model. These metrics are included for each prompt, and saved to a JSON file for each model per bias type (racial, gender). 

These results can be used to simply get the mean reward model score and the mean bias confidence score. _Remember to modify the filepath!_

## Running the Notebooks

All notebooks are run using an NVIDIA A100 GPU on Google Colab Pro. They require CUDA kernels.

## Results
The model responses to test data and the evaluation metrics for each response can be found in JSON files in the folder "Model Responses + Scores". The files are named as: `<model_type>_<bias_type>_response_scores.json`. 
