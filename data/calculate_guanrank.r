
source('guanrank.R')

cancer_all = c('coad','kirc','lihc')

for (cancer in cancer_all) {
    dat=read.table(paste0('feature_tcga_', cancer, '_short.tsv'),sep='\t',stringsAsFactors=F, quote='')
    # exclude time missing
    dat = dat[!is.na(as.numeric(dat[,2])),]
    dat[,2]=as.numeric(dat[,2])
    dat[,3]=as.numeric(dat[,3])
    # calculate
    mat = guanrank(dat[,2:3])
    label=rep(0,nrow(dat))
    label[order(dat[,2])] = mat[,3]
    # add guanrank column 
    dat_new=cbind(dat[,1:3],label,dat[,4:ncol(dat)])
    write.table(dat_new, file=paste0('feature_tcga_', cancer, '.tsv'), quote=F, sep='\t', row.names=F, col.names=F)
}


