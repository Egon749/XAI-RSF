args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(Sys.getenv("EW_RLIB"), .libPaths()))
library(survival)
d <- read.csv(args[1]); d$status <- factor(d$event, levels=0:2, labels=c("censor","target","other"))
fg <- finegray(Surv(t, status) ~ x1+x2, data=d, etype="target", timefix=FALSE)
cs <- data.frame(fgstart=0,fgstop=d$t,fgstatus=as.integer(d$event==1),fgwt=1,x1=d$x1,x2=d$x2)
evaluate <- function(tab,b) {
  x <- as.matrix(tab[,c("x1","x2")]); gradient <- colSums(x[tab$fgstatus==1,,drop=FALSE])
  for(u in sort(unique(tab$fgstop[tab$fgstatus==1]))) {
    keep <- tab$fgstart<u & tab$fgstop>=u
    w <- tab$fgwt[keep]*exp(x[keep,,drop=FALSE]%*%b)
    gradient <- gradient-sum(tab$fgstatus[tab$fgstop==u])*colSums(x[keep,,drop=FALSE]*as.numeric(w))/sum(w)
  }
  gradient/sum(tab$fgstatus)-0.1*b
}
b <- scan(args[2],quiet=TRUE)
write.table(rbind(evaluate(fg,b[1:2]),evaluate(cs,b[3:4])),args[3],row.names=FALSE,col.names=FALSE)
