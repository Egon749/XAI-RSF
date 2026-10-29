# Saved-reference verification only: no call to rfsrc/grow and no fitting.
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(Sys.getenv('EW_RLIB'),.libPaths()))
suppressPackageStartupMessages(library(randomForestSRC))
options(rf.cores=1)
items <- jsonlite::fromJSON(args[1])
result <- list()
for (i in seq_len(nrow(items))) {
  start <- proc.time()[['elapsed']]
  src <- items$path[i]
  model <- readRDS(file.path(src,'forest_PRIVATE.rds'))
  X <- read.csv(file.path(src,'predict_PRIVATE.csv'))
  pr <- predict(model,newdata=X,perf.type='none',importance=FALSE)
  original <- readBin(file.path(src,'cif_PRIVATE.bin'),'double',n=prod(dim(pr$cif)),size=8,endian='little')
  source_error <- max(abs(as.double(pr$cif)-original))
  path <- file.path(args[2],paste0(items$year[i],'_',items$seed[i],'_resaved_PRIVATE.rds'))
  stopifnot(!file.exists(path))
  saveRDS(model,path,compress=FALSE)
  reloaded <- predict(readRDS(path),newdata=X,perf.type='none',importance=FALSE)
  reload_error <- max(abs(pr$cif-reloaded$cif))
  minimum <- min(pr$cif); max_sum <- max(apply(pr$cif,c(1,2),sum))
  increment <- min(apply(pr$cif,c(1,3),diff))
  stopifnot(all(is.finite(pr$cif)),source_error<=1e-12,reload_error<=1e-12,minimum>=-1e-12,max_sum<=1+1e-12,increment>=-1e-12)
  result[[i]] <- data.frame(year=items$year[i],seed=items$seed[i],rows=nrow(X),native_times=length(pr$time.interest),source_prediction_error=source_error,reload_error=reload_error,min_cif=minimum,max_cause_sum=max_sum,min_increment=increment,new_fits=0,seconds=proc.time()[['elapsed']]-start)
}
write.csv(do.call(rbind,result),file.path(args[2],'standard_rsf_validation.csv'),row.names=FALSE)
