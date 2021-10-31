# Automated algorithm configurations on symbolic regression algorithms
Installation required: 
- R (version > 4.0)
- Python (version > 3.0)
- irace package -> https://iridia.ulb.ac.be/irace/
- Jupyter Notebook [optional]
- R studio [optional]

# User guide
Used tool for parameter tuning, with user guide cited: 
 - irace = https://cran.r-project.org/web/packages/irace/vignettes/irace-package.pdf 

Available algorithms and their API references in the respective directories:
 - GPLearn = https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-regressor
 - Random Forest Regressor [RandomForest] = https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor
 - Support Vector Regressor = https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html?highlight=svr#sklearn.svm.SVR

Other experimented algorithms are available at "Experimental" directory for future reference. 

Available list of text files for instances in scenario.txt:
 - i.txt = For Black-box problem datasets
 - Al.txt = For Al-Feynman's datasets [https://space.mit.edu/home/tegmark/aifeynman.html]
 - benchmark.txt & benchmarkTT.txt = For synthetically made datasets from GPLearn benchmarks [https://dl.acm.org/doi/pdf/10.1145/3205651.3208293]
 - St.txt = For Strogatz's datasets

Others: 
 - All datasets used and created are available at Datasets folder
 - Codes for algorithms without irace [ Random Search CV ] are available at "WithoutIrace" directory
 - Mean absolute error, Mean squared error, Root mean squared error and training time generated are added as .csv files at the "configuration" folder for each algorithm 
 - Logbooks for each algorithm on given datasets are available at Logbook folder
 - For visualisations and statistical reference on post-processing, access the "Post-processing.ipynb" in Jupyter Notebook to review.

# Operating System
- For MacOS and Windows users, install the irace package based on the given instructions in https://cran.r-project.org/web/packages/irace/vignettes/irace-package.pdf. If successful, it should be able to run using the command 
```irace```
- For Windows users, switch the target-runner in the scenario.txt to "./Unused/target-runner.bat" to run. 
