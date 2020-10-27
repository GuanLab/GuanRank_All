rm(list = ls())

library("randomForestSRC")
args=commandArgs(trailingOnly = TRUE)
#print(args[0])
#print(args[1])
#print(args[2])
#train_all = read.table(args[1])
#test_all = read.table(args[2])

train_all = read.table('train.dat')
test_all = read.table('test.dat')
test_data = test_all[,3:dim(test_all)[2]]
fit <- rfsrc(Surv(V1, V2)~., data = train_all, forest = TRUE, family = "surv")
res <- predict(fit, test_data)

print(res$predicted)

write(res$predicted, file = 'prediction.dat',sep='\n')


