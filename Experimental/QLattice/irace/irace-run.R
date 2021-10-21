setwd("Desktop/GPLearn/python/irace")
getwd()
load("irace.Rdata")

library(irace)

# Create the R objects scenario and parameters
parameters<-readParameters("parameters.txt")
scenario<-readScenario(filename="scenario.txt",scenario=defaultScenario())
irace(scenario= scenario,parameters= parameters)
testing.main(logFile = "./irace.Rdata")

iraceResults$irace.version
iraceResults$scenario$instances
head(iraceResults$scenario$testInstances)
repairConfiguration = function (configuration, parameters, digits)
{
  columns <- c("p1","p2","p3")
  # cat("Before"); print(configuration) configuration[columns] <- sort(configuration[columns]) # cat("After"); print(configuration) return(configuration)
}
best.config <- getFinalElites(iraceResults = iraceResults, n = 1) 

id <- best.config$.ID.
# Obtain the configurations using the identifier
# of the best configuration
all.exp <- iraceResults$experiments[,as.character(id)]
all.exp[!is.na(all.exp)]
# As an example, we get seed and instance of the experiments > # of the best candidate.
# Get index of the instances
pair.id <- names(all.exp[!is.na(all.exp)])
index <- iraceResults$state$.irace$instancesList[pair.id,"instance"] # Obtain the instance names
iraceResults$scenario$instances[index]
# Get the seeds
iraceResults$state$.irace$instancesList[index,"seed"]
iraceResults$experimentLog
results <- iraceResults$testing$experiments
results
iraceResults$testing$seeds
# Wilcoxon paired test
conf <- gl(ncol(results), # number of configurations 
           nrow(results), # number of instances
           labels = colnames(results))

pairwise.wilcox.test (as.vector(results), conf, paired = TRUE, p.adj = "bonf")
# Plot the results
configurationsBoxplot (results, ylab = "Solution cost")

# Get number of iterations
iters <- unique(iraceResults$experimentLog[, "iteration"])
# Get number of experiments (runs of target-runner) up to each iteration 
fes <- cumsum(table(iraceResults$experimentLog[,"iteration"]))
# Get the mean value of all experiments executed up to each iteration
# for the best configuration of that iteration.
elites <- as.character(iraceResults$iterationElites)
values <- colMeans(iraceResults$experiments[, elites])
dev.new(width=100, height=50, unit="px")
plot(fes, values, type = "s",
     xlab = "Number of runs of the target algorithm",
     ylab = "Mean value over testing set\n\n")
points(fes, values)
text(fes, values, elites, pos = 1)

parameterFrequency(iraceResults$allConfigurations, iraceResults$parameters)
iraceResults$iterationElites

ablation(iraceLogFile = "irace.Rdata", pdf.file = "plot-ablation.pdf")
# Execute all elite configurations in the iterations
psRace(iraceLogFile="irace.Rdata", elites=TRUE)
# Execute a set of configurations IDs providing budget
iraceResults$allConfigurations
psRace(iraceLogFile="irace.Rdata", elites=TRUE)
psRace(iraceLogFile="irace.Rdata",conf.ids = c(34,60), max.experiments=500)

