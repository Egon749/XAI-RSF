args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(Sys.getenv('EW_RLIB'), .libPaths()))
suppressPackageStartupMessages(library(randomForestSRC))
suppressPackageStartupMessages(library(survival))
options(rf.cores=1)
cfg <- jsonlite::fromJSON(args[1])
tr <- read.csv(cfg$train)
te <- read.csv(cfg$predict)
samp <- as.matrix(read.csv(cfg$inbag, header=FALSE))
fit <- rfsrc(Surv(time,status)~., data=tr, ntree=cfg$trees,
             mtry=cfg$mtry, nodesize=cfg$nodesize, nodedepth=cfg$depth,
             nsplit=cfg$nsplit, splitrule='logrankCR', cause=c(1,1),
             bootstrap='by.user', samp=samp, samptype='swr',
             ntime=0, perf.type='none', importance=FALSE,
             seed=-as.integer(cfg$seed+3000), forest=TRUE, save.memory=TRUE)
stopifnot(fit$family=='surv-CR', identical(sort(as.numeric(fit$event.info$event.type)),c(1,2)))
pr <- predict(fit, newdata=te, perf.type='none', importance=FALSE)
con <- file(file.path(cfg$output,'cif.bin'),'wb')
writeBin(as.double(pr$cif),con,size=8,endian='little'); close(con)
write.table(pr$time.interest,file.path(cfg$output,'times.csv'),row.names=FALSE,col.names=FALSE,sep=',')
saveRDS(fit,file.path(cfg$output,'forest_PRIVATE.rds'),compress=FALSE)
jsonlite::write_json(list(version=as.character(packageVersion('randomForestSRC')),
  survival_version=as.character(packageVersion('survival')),R=R.version.string,
  dims=dim(pr$cif),leaf_counts=fit$leaf.count,
  bootstrap='by.user, identical externally generated whole-group inbag counts',
  splitrule=fit$splitrule,cause=c(1,1)),file.path(cfg$output,'r_meta.json'),auto_unbox=TRUE)
