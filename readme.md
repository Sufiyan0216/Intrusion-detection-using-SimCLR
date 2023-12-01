###  This is a project that 

+ replicates an exisiting method that utilize SimCLR on intrusion detection
+ replace the backbone network of the extractor to MLP

In the files of experiment-\<model-name\>.ipynb,  we hyperparameter searched the best params to be evaluate the performance

In the files of train-\<model-name\>.ipynb, we trained the models with best params and saved the preprocessor and the models along with 5-step-wise checkpoints. 

In the files of eval.ipynb, we evaluate the resnet based extractor(original) and the MLP based extractor(ours) on a set of classic ML classifiers. 

All the other files are clearly named the intent so no more introduction to those files. 

### Conclusion 

We found

+ It’s possible to use an extractor with a simpler backbone 
  network to substitute. 
+ The extractor with a simpler backbone network perform 
  similar when on non-linear learning methods and not-
  learning-based methods while perform worse on linear 
  learning methods.
+ It significantly(^10) reduces the computation resources 
  needed by a feature extractor.