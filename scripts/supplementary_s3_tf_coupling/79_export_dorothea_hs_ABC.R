
args <- commandArgs(trailingOnly=TRUE)
out <- if(length(args)>=1) args[[1]] else
"<TABULA_PROJECT_ROOT>/corrected_full_analysis_v2/79_78_1_CORRECTION_AND_TF_DISCOVERY_20261004/00_INPUTS/79_DOROTHEA_HS_ABC.csv"
dir.create(dirname(out),recursive=TRUE,showWarnings=FALSE)
if(!requireNamespace("dorothea",quietly=TRUE)){
  cat("dorothea not installed; export skipped\n"); quit(status=2)
}
suppressPackageStartupMessages(library(dorothea))
data("dorothea_hs",package="dorothea",envir=environment())
net <- get("dorothea_hs",envir=environment())
tfcol <- if("source"%in%colnames(net)) "source" else if("tf"%in%colnames(net)) "tf" else stop("No TF column")
x <- data.frame(source=toupper(as.character(net[[tfcol]])),
                target=toupper(as.character(net$target)),
                mor=as.numeric(net$mor),
                confidence=as.character(net$confidence))
x <- x[x$confidence %in% c("A","B","C") & is.finite(x$mor) & x$source!="" & x$target!="",]
x <- x[!duplicated(x[,c("source","target")]),]
write.csv(x,out,row.names=FALSE)
cat("DOROTHEA_EXPORT_COMPLETE rows=",nrow(x)," TFs=",length(unique(x$source)),"\n")
