# US census data mini-project

### A brief summary

Please find all code and config files in `python_work`. This includes several notebooks (numbered in order), two json files, and two python files with some quickly-written helper classes.

Given the short timeframe, notebooks were used for EDA and training, with more complex code being placed in the two `*.py` files. It makes me uncomfortable to not have tests for these (the preprocessor, in particular, would probably have benefitted from a TDD approach), but it hopefully gives a little flavour. If this stuff was quick and easy, DSS wouldn't be as helpful :-)

### Reproducing results

Apologies for not including a `requirements.txt` -- this was done in an existing (overcomplicated) environment. It should be sufficient to have installed (with dependencies):
* pandas
* scikit-learn
* matplotlib
* xgboost
* jupyter

Before running the code, please also set an environment variable `DATA_FOLDER`, being the path to the folder containing all the data files. All training code is seeded (though for running a final evaluation on the "test" set, cell execution order does matter a little - again, time constraints)