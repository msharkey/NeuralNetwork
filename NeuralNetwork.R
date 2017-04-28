

#install.packages("XLConnect")
#install.packages("RSNNS")

library("XLConnect")
library("RSNNS")



#Data Import----------------------------------------------------------------------------------------------------
setwd("C:\\Users\\mshar\\")
file <- file.path("downloads", "Sepsis-data.xls")
my.trainset <- readWorksheetFromFile(file, sheet = 1, startRow = 1)
my.testset <- readWorksheetFromFile(file, sheet = 2, startRow = 1)

my.trainset <- my.trainset[sample(1:nrow(my.trainset),length(1:nrow(my.trainset))),1:ncol(my.trainset)]

my.trainset.values <- my.trainset[, 1:18]
my.trainset.targets <- my.trainset[,19]

my.testset <- my.testset[sample(1:nrow(my.testset), length(1:nrow(my.testset))), 1:ncol(my.testset)]

my.testset.values <- my.testset[, 1:18]
my.testset.targets <- my.testset[, 19]

#--------------------------------------------------------------------------------------------------------------


#Model--------------------------------------------------------------------------------------------------------
my.getdate <- Sys.time()
my.resultsReport <- 0
my.iterations <- 10000
my.batchNum <-0
for (i in 1:my.iterations) { 

  #Set Network Configurations
  #----------------------------------------------------
  my.maxit <- sample(500:2000, 1)
    my.learningrate <- runif(1, .1, .5)
    my.tolerance <- runif(1, .08, .25)
  my.numhiddenlayers <- sample(1:2,1)   
  my.size <- ifelse(my.numhiddenlayers == 1, my.hiddenLayerArch <- (sample(4:15, 1)), my.hiddenLayerArch <- c((sample(4:12, 1)), (sample(2:6, 1))))
  my.learnfunctionswitch <-sample(1:8,1)
  my.learnfunction <- switch(my.learnfunctionswitch, "Std_Backpropagation", "BackpropBatch","BackpropChunk","BackpropMomentum","BackpropWeightDecay","Rprop","Quickprop","SCG")
  #----------------------------------------------------
  
  my.executiontime <- system.time(

    my.model <- mlp(x = my.trainset.values, y = my.trainset.targets,
                    size = my.hiddenLayerArch,                 
                    maxit = my.maxit,
                    initFunc = "Randomize_Weights",
                    initFuncParams = c(-0.3, 0.3),
                    learnFunc = my.learnfunction,
                    learnFuncParam = c(my.learningrate, my.tolerance),
                    updateFunc = "Topological_Order",
                    updateFuncParams = c(0),
                    hiddenActFunc = "Act_Logistic",
                    shufflePatterns = TRUE,
                    lineOut = TRUE,
                    inputsTest = my.testset.values,
                    targetsTest = my.testset.targets
                    
    )
    
  )
  
  #Get elapsed time 
  my.executiontime <- as.vector(my.executiontime)
  
  
  #Get MSE
  
  my.train.mse <- (1 / nrow(my.trainset.values)) * sum((my.model$fitted.values - my.trainset.targets) ^ 2)
  rsnns.train.rmse <- sqrt(my.train.mse)
  
  my.test.mse <- (1 / nrow(my.testset.values)) * sum((my.model$fittedTestValues - my.testset.targets) ^ 2)
  rsnns.test.rmse <- sqrt(my.test.mse)
  
  
  #Get Sensitivity and Specificity 
  
  # Training Data
  my.trainprediction<- predict(my.model, my.trainset.values)
  my.trainprediction <- ifelse(my.trainprediction >= .9, 1, 0)
  
  my.outputtrain <-cbind(my.trainprediction, my.trainset.targets)
  
  my.trainSensitivity <- sum(my.outputtrain[my.trainset.targets == 1, 1]) / sum(my.outputtrain[my.trainset.targets == 1, 2])
  my.trainSpecificty <- ( sum(ifelse(my.outputtrain[my.trainset.targets == 0, 1] == 0, 1, 0)) )/ (nrow(my.outputtrain) - sum(my.outputtrain[my.trainset.targets == 1, 2]))
  
  #Testing Data
  my.testprediction <- predict(my.model, my.testset.values)
  my.testprediction <- ifelse(my.testprediction >= .9, 1, 0)
  
  my.outputtest <- cbind(my.testprediction, my.testset.targets)
  
  my.testSensitivity <- sum(my.outputtest[my.testset.targets == 1, 1]) / sum(my.outputtest[my.testset.targets == 1, 2])
  my.testSpecificty <- (sum(ifelse(my.outputtest[my.testset.targets == 0, 1] == 0, 1, 0))) / (nrow(my.outputtest) - sum(my.outputtest[my.testset.targets == 1, 2]))
  
  
  
    my.results <- cbind(batch = my.batchNum,
                      iterations = my.maxit,
                      hiddenlayerneuronsL1 = my.hiddenLayerArch[1],
                      hiddenlayerneuronsL2 = my.hiddenLayerArch[2],
                       LearningFunction = my.learnfunction,
                      NumberofHiddenLayers = my.numhiddenlayers,
                      learningrate = my.learningrate,
                      Tolerance = my.tolerance,
                      trainingmse = rsnns.train.rmse,
                      testmse = rsnns.test.rmse,
                      excutiontime = my.executiontime[3],
                      TrainingSensitivity = my.trainSensitivity,
                      TrainingSpecificty = my.trainSpecificty,
                      TestingSensitivity = my.testSensitivity,
                      TestingSpecificty = my.testSpecificty)



my.resultsReport <- rbind(my.resultsReport, my.results) 

#Write results to CSV, each iteration truncates CSV, incremental load pattern is desired but I'm running out of time, so I'm taking a performance inhibitng shortcut, consider rewirite for production
write.csv(my.resultsReport, file = "NetworkTesting.csv")

my.batchNum <- my.batchNum + 1
}



#FinalModel--------------------------------------------------------------------------------------------------------
my.getdate <- Sys.time()
my.resultsReport <- 0
my.iterations <- 1
my.batchNum <- 0
for (i in 1:my.iterations) {
    #iterations hiddenlayerneuronsL1 hiddenlayerneuronsL2 LearningFunction NumberofHiddenLayers learningrate Tolerance
    #1297 9 2 SCG 2 0.17420436 0.117078303


    #Set Network Configurations
    #----------------------------------------------------
    my.maxit <- 1450 #sample(500:2000, 1)
    my.learningrate <- .44 #runif(1, .1, .5)
    my.tolerance <- .10 #runif(1, .08, .25)
    my.numhiddenlayers <- 1 # sample(1:2, 1)
    my.size <- c(11) #ifelse(my.numhiddenlayers == 1, my.hiddenLayerArch <- (sample(4:15, 1)), my.hiddenLayerArch <- c((sample(4:12, 1)), (sample(2:6, 1))))
    my.learnfunctionswitch <- sample(1:8, 1)
    my.learnfunction <- "Quickprop" #switch(my.learnfunctionswitch, "Std_Backpropagation", "BackpropBatch", "BackpropChunk", "BackpropMomentum", "BackpropWeightDecay", "Rprop", "Quickprop", "SCG")
    #----------------------------------------------------

    my.executiontime <- system.time(

    my.model <- mlp(x = my.trainset.values, y = my.trainset.targets,
                    size = my.hiddenLayerArch,
                    maxit = my.maxit,
                    initFunc = "Randomize_Weights",
                    initFuncParams = c(-0.3, 0.3),
                    learnFunc = my.learnfunction,
                    learnFuncParam = c(my.learningrate, my.tolerance),
                    updateFunc = "Topological_Order",
                    updateFuncParams = c(0),
                    hiddenActFunc = "Act_Logistic",
                    shufflePatterns = TRUE,
                    lineOut = TRUE,
                    inputsTest = my.testset.values,
                    targetsTest = my.testset.targets

    )

    )

    #Get elapsed time 
    my.executiontime <- as.vector(my.executiontime)


    #Get MSE

    my.train.mse <- (1 / nrow(my.trainset.values)) * sum((my.model$fitted.values - my.trainset.targets) ^ 2)
    rsnns.train.rmse <- sqrt(my.train.mse)

    my.test.mse <- (1 / nrow(my.testset.values)) * sum((my.model$fittedTestValues - my.testset.targets) ^ 2)
    rsnns.test.rmse <- sqrt(my.test.mse)


    #Get Sensitivity and Specificity 

    # Training Data
    my.trainprediction <- predict(my.model, my.trainset.values)
    my.trainprediction <- ifelse(my.trainprediction >= .9, 1, 0)

    my.outputtrain <- cbind(my.trainprediction, my.trainset.targets)
    #my.outputtest


    my.trainSensitivity <- sum(my.outputtrain[my.trainset.targets == 1, 1]) / sum(my.outputtrain[my.trainset.targets == 1, 2])
    my.trainSpecificty <- (sum(ifelse(my.outputtrain[my.trainset.targets == 0, 1] == 0, 1, 0))) / (nrow(my.outputtrain) - sum(my.outputtrain[my.trainset.targets == 1, 2]))

    #Testing Data
    my.testprediction <- predict(my.model, my.testset.values)
    my.testprediction <- ifelse(my.testprediction >= .9, 1, 0)

    my.outputtest <- cbind(my.testprediction, my.testset.targets)

    my.testSensitivity <- sum(my.outputtest[my.testset.targets == 1, 1]) / sum(my.outputtest[my.testset.targets == 1, 2])
    my.testSpecificty <- (sum(ifelse(my.outputtest[my.testset.targets == 0, 1] == 0, 1, 0))) / (nrow(my.outputtest) - sum(my.outputtest[my.testset.targets == 1, 2]))



    my.results <- cbind(
    batch = my.batchNum,
                      iterations = my.maxit,
                      hiddenlayerneuronsL1 = my.hiddenLayerArch[1],
                      hiddenlayerneuronsL2 = my.hiddenLayerArch[2],
                       LearningFunction = my.learnfunction,
                      NumberofHiddenLayers = my.numhiddenlayers,
                      trainingmse = rsnns.train.rmse,
                      testmse = rsnns.test.rmse,
                      excutiontime = my.executiontime[3],
                      TrainingSensitivity = my.trainSensitivity,
                      TrainingSpecificty = my.trainSpecificty,
                      TestingSensitivity = my.testSensitivity,
                      TestingSpecificty = my.testSpecificty)



    my.resultsReport <- rbind(my.resultsReport, my.results)



    my.batchNum <- my.batchNum + 1
}

my.resultsReport







