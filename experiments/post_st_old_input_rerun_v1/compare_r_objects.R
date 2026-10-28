# RDS bytes include run timing and closure environments; audit tree payload exactly.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2)
manifest <- read.csv(args[1],stringsAsFactors=FALSE)
rows <- lapply(seq_len(nrow(manifest)),function(i){
  a <- readRDS(manifest$old_path[i]); b <- readRDS(manifest$new_path[i])
  top <- union(names(a),names(b))
  different <- top[!vapply(top,function(k)identical(a[[k]],b[[k]]),logical(1))]
  fa <- a$forest$sampsize; fb <- b$forest$sampsize
  function_code_exact <- identical(body(fa),body(fb)) && identical(formals(fa),formals(fb))
  a$forest$sampsize <- NULL; b$forest$sampsize <- NULL
  forest_payload_exact <- identical(a$forest,b$forest)
  for(k in c('ctime.internal','ctime.external')){a[[k]] <- NULL;b[[k]] <- NULL}
  result <- data.frame(test_year=manifest$test_year[i],seed=manifest$seed[i],
    differing_top_level_fields=paste(different,collapse=';'),
    sampsize_function_body_formals_exact=function_code_exact,
    forest_payload_exact_excluding_sampsize_closure=forest_payload_exact,
    full_object_exact_excluding_timing_and_sampsize_closure=identical(a,b))
  stopifnot(function_code_exact,forest_payload_exact,identical(a,b))
  result
})
write.csv(do.call(rbind,rows),args[2],row.names=FALSE)
cat('Six standard-RSF forests: exact parameters, data and prediction payloads; RDS byte differences are timing/closure serialization.\n')
