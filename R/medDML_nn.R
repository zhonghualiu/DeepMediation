
# function for estimation, se, and p-values
medDML_ann=function(y,d,m,x,trim=0.05, order=1, multmed=TRUE, fewsplits=FALSE,hyper_nn=hyper_nn){
  if (multmed!=0) temp=hdmedalt_nn(y=y,d=d,m=m,x=x, trim=trim, hyper_nn=hyper_nn)
  ATE = temp$ATE
  val_loss = temp$val_loss
  val_loss_min = temp$val_loss_min
  val_epoch = temp$val_epoch


  eff=ATE[1:6]
  se=sqrt( (ATE[7:12])/ATE[13])
  results=rbind(eff,se, 2*pnorm(-abs(eff/se)))
  colnames(results)=c("total", "dir.treat", "dir.control", "indir.treat", "indir.control", "Y(0,M(0))")
  rownames(results)=c("effect","se","p-val")
  ntrimmed=length(d)-ATE[13]

  list("results"=results, "ntrimmed"=ntrimmed, "val_loss"=val_loss,
       "val_loss_min"=val_loss_min, "val_epoch"=val_epoch)

}


