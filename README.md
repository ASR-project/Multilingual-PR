# Multilingual-PR

Implementation of the project ```Multi-lingual Phoneme Recognition using self-supervised methods on foreign languages```

## TO DO 

### Data processing part

- [ ] Explore the dataset on Mozilla common voices
- [ ] Understand how phoible works
- [ ] Get the features from a pre-trained model on HuggingFace on the retrieved dataset
- [ ] Split the dataset into a trainval / test set. Make sure that the speakers do not occur both on te train set and test set

### Modeling part

- [ ] Implement CTC algorithm using PyTorch
- [ ] Train the model on the built dataset

### Evaluation

- [ ] Evaluate the model on test custom test set built at the step 1
