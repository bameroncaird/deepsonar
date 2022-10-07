This repo is an implementation of the paper DeepSonar: Towards Effective and Robust Detection of AI-Synthesized Fake Voices. The paper can be found at this link: https://arxiv.org/abs/2005.13770v3. I am not an author of the paper, nor do I have any affiliation with the authors. My research group simply wanted to use this model to run some research experiments because it is, to our knowledge, the top performing fake voice detector as of today (October 2022). All credit for the ideas in this project goes to the original authors of DeepSonar, e.g., Wang et al. However, the code for this model was not released and the authors did not respond to a request for the code, so we will implement this model ourselves.

You can find the venv configuration in the requirements.txt file (although I will probably be deleting this file and doing this in a better way).

I am in the process of updating this repo.

In the old_code/ folder, you can find my initial implementation from when I was in undergrad. I don't like the structure, organization, or implementation from this approach, which is why I'm updating the repo.

In the src/ folder, you can find the code implementation for this project.

In the sr_models/ folder, you can find different pre-trained models for speaker recognition (SR). This is because one of DeepSonar's key contributions is monitoring the layer-wise neuron behaviors of a DNN-based SR system in order to classify the real and fake voices.

A crucial part of this project, as mentioned in the last point, is the DNN-based SR system. The code for the SR system used in the original DeepSonar paper can be found at this link: https://github.com/WeidiXie/VGG-Speaker-Recognition. All credit for this model and code goes to the corresponding authors Xie et al. Their paper is called Utterance-Level Aggregation for Speaker Recognition in the Wild and can be found at this link: https://arxiv.org/abs/1902.10107.