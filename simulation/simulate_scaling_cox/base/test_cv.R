source('score.R');

a<-read.table('input.txt',sep="\t", header=FALSE, row.names=NULL);
y <- score_q1a(a[,1],a[,2],a[,3])
c=y$cIndex
write(c,file='cIndex.txt')
