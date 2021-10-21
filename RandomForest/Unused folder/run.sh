# create a directory for saving output
mkdir -p results

# example of using irace2pyimp with filter conditions and normalisation:
# - filter tuning data such that only configurations with n_templates_middle<=40 are used, then convert irace.Rdata to PIMP's input format,
# - the cost values of all configurations are normalised on an instance-basis
# irace2pyimp --out-dir results --irace-data-file irace.Rdata

# example 1: convert irace.Rdata to PyImp's input format, without normalisation
# irace2pyimp --out-dir results --irace-data-file irace.Rdata

# example 2: convert irace.Rdata to PyImp's input format, without normalisation, only configurations with algorithm!='as' are taken
#irace2pyimp --out-dir results --instance-feature-file features.csv --filter-conditions "algorithm!='as'" --irace-data-file irace.Rdata

# example 3: convert irace.Rdata to PyImp's input format, with normalisation based on instance
# irace2pyimp --out-dir results --instance-feature-file features.csv --normalise instance --irace-data-file irace.Rdata

# example 4: convert irace.Rdata to PyImp's input format, with normalisation based on feature-vector values
# irace2pyimp --out-dir results --instance-feature-file features.csv --normalise feature --irace-data-file irace.Rdata

# call PyImp (executable name is "pimp") with all modules (fanova, abalation analysis, and forward selection)
cd results
pimp -S scenario.txt -H runhistory.json -T traj_aclib2.json -M all
cd ..
