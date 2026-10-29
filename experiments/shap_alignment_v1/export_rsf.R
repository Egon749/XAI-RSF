# Read an existing randomForestSRC forest, preserving native splits and counts.
# No call to rfsrc/grow: get.tree predictions recover its existing terminal CIFs.
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(Sys.getenv('EW_RLIB'), .libPaths()))
suppressPackageStartupMessages(library(randomForestSRC))
options(rf.cores=1)
cfg <- jsonlite::fromJSON(args[1])
stopifnot(!file.exists(cfg$output))
dir.create(cfg$output, recursive=TRUE)
fit <- readRDS(cfg$model)
stopifnot(fit$family=='surv-CR', as.character(packageVersion('randomForestSRC'))=='3.9.0')
stopifnot(all(fit$forest$nativeArray$mwcpSZ==0), !anyNA(fit$xvar))
q <- matrix(readBin(cfg$query, 'double', n=cfg$n_query*cfg$p, size=8, endian='little'),
            ncol=cfg$p, byrow=TRUE)
colnames(q) <- fit$xvar.names
native <- fit$forest$nativeArray
splitrows <- which(native$parmID>0)
splitrows <- head(splitrows,64)
probes <- matrix(0, length(splitrows)*3L, ncol(q))
for (r in seq_along(splitrows)) {
  a <- native[splitrows[r],]
  for (u in 1:3) {
    row <- as.double(fit$xvar[1,]);cut <- a$contPT
    row[a$parmID] <- cut+(u-2)*.Machine$double.eps*max(1,abs(cut))
    probes[(r-1)*3+u,] <- row
  }
}
allx <- rbind(as.matrix(fit$xvar), q, probes)
allx <- as.data.frame(allx)
trees <- list()
preds <- matrix(0, nrow(allx), fit$ntree)
membership <- matrix(0L, nrow(allx), fit$ntree)
max_within_leaf <- 0
for (b in seq_len(fit$ntree)) {
  a <- fit$forest$nativeArray[fit$forest$nativeArray$treeID==b,,drop=FALSE]
  pr <- predict(fit, newdata=allx, get.tree=b, membership=TRUE,
                perf.type='none', importance=FALSE)
  ti <- findInterval(cfg$horizon_days, pr$time.interest)
  stopifnot(ti>0)
  y <- pr$cif[,ti,1]
  ids <- pr$membership[,b]
  leaf <- a$parmID==0
  stopifnot(setequal(ids, a$nodeID[leaf]))
  val <- rep(0, nrow(a))
  for (i in which(leaf)) {
    vv <- y[ids==a$nodeID[i]]
    stopifnot(length(vv)>0, max(vv)-min(vv)<1e-14)
    max_within_leaf <- max(max_within_leaf, max(vv)-min(vv))
    val[i] <- vv[1]
  }
  trees[[b]] <- list(feature=as.integer(a$parmID)-1L, threshold=a$contPT,
                    cover=a$nodeSZ, leaf_id=a$nodeID, values=val)
  # jsonlite digits=NA still rounds some binary64 values; never use JSON cuts.
  writeBin(as.double(a$contPT),file.path(cfg$output,sprintf('tree_%03d_cuts_PRIVATE.bin',b)),size=8,endian='little')
  writeBin(as.double(val),file.path(cfg$output,sprintf('tree_%03d_values_PRIVATE.bin',b)),size=8,endian='little')
  preds[,b] <- y
  membership[,b] <- ids
}
pr <- predict(fit,newdata=allx,perf.type='none',importance=FALSE)
ti <- findInterval(cfg$horizon_days,pr$time.interest)
original <- pr$cif[,ti,1]
err <- max(abs(rowMeans(preds)-original))
stopifnot(err<1e-14)
jsonlite::write_json(list(trees=trees,ntree=fit$ntree,p=ncol(allx),
  horizon_days=cfg$horizon_days,feature_names=fit$xvar.names,n_train=nrow(fit$xvar),
  n_query=nrow(q),n_boundary=nrow(probes),
  float_payload='little-endian binary64; JSON float arrays are descriptive only',
  new_fits=0,aggregation_error=err,max_within_leaf_error=max_within_leaf,
  R=R.version.string,randomForestSRC=as.character(packageVersion('randomForestSRC'))),
  file.path(cfg$output,'forest_export_PRIVATE.json'),auto_unbox=TRUE,digits=NA)
writeBin(as.double(t(as.matrix(allx))),file.path(cfg$output,'allx_PRIVATE.bin'),size=8,endian='little')
writeBin(as.double(original),file.path(cfg$output,'original_prediction_PRIVATE.bin'),size=8,endian='little')
writeBin(as.integer(t(membership)),file.path(cfg$output,'membership_PRIVATE.bin'),size=4,endian='little')
cat(jsonlite::toJSON(list(status='COMPLETE',trees=fit$ntree,new_fits=0,
  aggregation_error=err),auto_unbox=TRUE),'\n')
