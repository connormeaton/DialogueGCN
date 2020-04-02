## DialogueGCN

This repo contains an implementation of [DialogueGCN][1], a graph convolutional neural network for emotion recognition in conversation. It was built by Ghosal et al. (2019) and trained on the IEMOCAP dataset. I am using this model as a guide to construct a graph convolutional model on local data.

### Updates:

- 4/02/20: Working on changing input data to a local source. Progress has been difficult:
	- See [this][2] repo for my progress on the CNN-based text feature generation required to transform input text into the proper modelling format.
	- From my understanging, input data should be an np.array shape `[n, n, 100]`
- 3/31/19: Original model has been trained on IEMOCAP data. It achieved the reported 64% test accuracy. It is saved under `saved_model.pt`.

### Questions:
- How should text-to-sequencing happen? Should hierarchical structure of conversations be preserved somehow? The structure text looks like this:


		                        |    corpus    |
                               /        |       \
                              c1       c2       c3
                            / | \     / | \    / | \
                           u1 u2 u3 u1 u2 u3  u1 u2 u3

	Or, should structure be flattened, to look like this?:


                            |        corpus        |
                            / | \     / | \    / | \
                           u1 u2 u3 u4 u5 u6  u7 u8 u9
                
	Not sure if this is the best approach...
        loosing conversation level granularity, but context may not exceed window function?
                
  [1]: https://arxiv.org/pdf/1908.11540.pdf
  [2]: https://github.com/cmeaton/CNN_for_text_features
