rm(list = ls())
library('glmnet')
train_all<-read.table('train.dat')
test_all<-read.table('test.dat');
train_data<-train_all[,3:dim(train_all)[2]]
test_data<-test_all[,3:dim(test_all)[2]]
print(test_data)
train_time<-train_all[,1];
train_status<-train_all[,2];
test_time<-test_all[,1];
test_status<-test_all[,2];
y=cbind(time=train_time,status=train_status)
fit=glmnet(y, x=as.matrix(train_data), family="cox", lambda=1.0,alpha=0.05)
#fit=glmnet(y, x=as.matrix(train_data), family="cox")#, lambda=1.0,alpha=0.05)
result<-predict(fit,as.matrix(test_data),s=1.0,type="response")

write(result,file='prediction.dat',sep='\n')
