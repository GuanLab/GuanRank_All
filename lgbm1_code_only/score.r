library("survcomp")

args = commandArgs(TRUE)
cindex_dataframe = read.csv(args[1], sep="\t")
cindex = concordance.index(x = cindex_dataframe$pred, surv.time = cindex_dataframe$time, surv.event = cindex_dataframe$status)
output = cindex[1]
output





