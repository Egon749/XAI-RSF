from pathlib import Path
import argparse,os,subprocess
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import sha,write


def run(root):
    script=root/'audit/check_training_AJ.R'
    script.write_text('''args <- commandArgs(TRUE)
.libPaths(c(Sys.getenv("EW_RLIB"),.libPaths()))
library(survival)
d <- read.csv(args[1]); ans <- list()
for (part in split(d,paste(d$cutoff_year,d$cohort,sep="_"))) {
  causes <- sort(unique(part$event[part$event>0]))
  if(length(causes)==0) {
    ans[[length(ans)+1]] <- data.frame(cutoff_year=part$cutoff_year[1],cohort=part$cohort[1],horizon_months=c(6,12,24),termination=0,recovery=0)
    next
  }
  part$status <- factor(part$event,levels=c(0,causes),labels=c("censor",c("one","two")[causes]))
  fit <- survfit(Surv(time,status)~1,data=part,timefix=FALSE)
  s <- summary(fit,times=c(6,12,24),extend=TRUE)
  take <- function(name) if(name %in% colnames(s$pstate)) s$pstate[,name] else rep(0,3)
  ans[[length(ans)+1]] <- data.frame(cutoff_year=part$cutoff_year[1],cohort=part$cohort[1],horizon_months=c(6,12,24),
    termination=take("one"),recovery=take("two"))
}
write.csv(do.call(rbind,ans),args[2],row.names=FALSE)
cat(as.character(packageVersion("survival")))
''')
    result=subprocess.run([os.environ['CRFOREST_RSCRIPT'],'--vanilla',str(script),str(root/'private/aj_check_input_PRIVATE.csv'),str(root/'audit/R_training_AJ.csv')],capture_output=True,text=True)
    (root/'audit/R_training_AJ.log').write_text(result.stdout+'\n'+result.stderr);assert result.returncode==0,result.stderr
    left=pd.read_csv(root/'reports/cohort_aj.csv');right=pd.read_csv(root/'audit/R_training_AJ.csv')
    merged=left.merge(right,on=['cutoff_year','cohort','horizon_months'],validate='one_to_one');assert len(merged)==len(left)
    eligible=merged.horizon_supported
    a=merged.loc[eligible,['aj_termination','aj_recovery']].to_numpy();b=merged.loc[eligible,['termination','recovery']].to_numpy()
    error=float(abs(a-b).max());assert error<1e-12
    assert merged.loc[~eligible,['aj_termination','aj_recovery']].isna().all().all()
    write(root/'audit/step0_independent_R_check.json',dict(status='PASS',survival_version=result.stdout.strip(),max_error=error,
        supported_cohort_horizons=int(eligible.sum()),unsupported_horizons_explicitly_NA=int((~eligible).sum()),
        script_sha256=sha(script),no_test_data_read=True))
    print('Independent training AJ check passed; max error',error)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
