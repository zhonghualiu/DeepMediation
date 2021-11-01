
# k=3; trim=0.05; order=1; i =1;fewsplits=FALSE
# units=20; layers=1; l1=0; epochs=500; batch_size=128

hdmedalt_nn=function(y,d,m,x, trim=0.05,batch_size=100,hyper_nn=hyper_nn){

  l1_1 =hyper_nn[1,1]; l1_2 =hyper_nn[1,2]; l1_3 =hyper_nn[1,3]; l1_4 =hyper_nn[1,4]; l1_5 =hyper_nn[1,5]; l1_6 =hyper_nn[1,6]; l1_7 = hyper_nn[1,7];l1_8 = hyper_nn[1,8];
  layer_1 =hyper_nn[2,1]; layer_2 =hyper_nn[2,2]; layer_3 =hyper_nn[2,3]; layer_4 =hyper_nn[2,4]; layer_5 =hyper_nn[2,5]; layer_6 =hyper_nn[2,6]; layer_7 = hyper_nn[2,7];layer_8 = hyper_nn[2,8];
  units_1 =hyper_nn[3,1]; units_2 =hyper_nn[3,2]; units_3 =hyper_nn[3,3]; units_4 =hyper_nn[3,4]; units_5 =hyper_nn[3,5]; units_6 =hyper_nn[3,6]; units_7 = hyper_nn[3,7];units_8 = hyper_nn[3,8];
  epochs_1 =hyper_nn[4,1]; epochs_2 =hyper_nn[4,2]; epochs_3 =hyper_nn[4,3]; epochs_4 =hyper_nn[4,4]; epochs_5 =hyper_nn[4,5]; epochs_6 =hyper_nn[4,6]; epochs_7 = hyper_nn[4,7];epochs_8 = hyper_nn[4,8];

  #----------------------------------------------------------------------

  ybin=1*(length(unique(y))==2 & min(y)==0 & max(y)==1)
  #generate higher order terms for lasso
  xm=cbind(x,m)

  stepsize=ceiling((1/3)*length(d))
  nobs = min(3*stepsize,length(d)); set.seed(1); idx = sample(nobs);
  sample1 = idx[1:stepsize]; sample2 = idx[(stepsize+1):(2*stepsize)];
  sample3 = idx[(2*stepsize+1):nobs];

  y1m0=c();y1m1=c();y0m0=c(); y0m1=c(); selall=c()
  # crossfitting procedure that splits sample in training an testing data
  val_epoch = val_loss_min = val_loss = val_t_mse = val_t_bias = numeric()
  for (i in 1:3){
    if (i==1) {tesample=sample1; musample=sample2; deltasample=sample3}
    if (i==2) {tesample=sample3; musample=sample1; deltasample=sample2}
    if (i==3) {tesample=sample2; musample=sample3; deltasample=sample1}
    trsample=c(musample,deltasample); dte=d[tesample]; yte=y[tesample]
    # in case that fewsplits is one, psample and musample are merged

    x=as.matrix(x,nrow(x),ncol(x)); xm=as.matrix(xm,nrow(xm),ncol(xm))

    # 1. fit Pr(D=1|M,X) in total of training data
    pmx=nn_bin(d[trsample],xm[trsample,],
               d[tesample],xm[tesample,], units= units_1, layers=layer_1,
               epochs=epochs_1, batch_size=batch_size, l1= l1_1 )
    pmx_w=pmx$w;
    pmxte=pmx$y_pred


    pmx_loss=pmx$val_loss
    pmx_loss_min = min(pmx$val_loss_all)
    pmx_epoch = which.min(pmx$val_loss_all)


    # 2. fit Pr(D=1|X) in total of training data
    px=nn_bin(d[trsample],x[trsample,],
              d[tesample],x[tesample,], units= units_2 ,layers=layer_2,
              epochs=epochs_2, batch_size=batch_size, l1= l1_2)
    pxte = px$y_pred
    px_w=px$w


    px_loss=px$val_loss
    px_loss_min = min(px$val_loss_all)
    px_epoch = which.min(px$val_loss_all)

    if (ybin!=1){
      # 3. fit E(Y|M,X,D=1) in first training data
      eymx1 = nn_reg(y[musample[d[musample]==1]],xm[musample[d[musample]==1],],
                     y[c(tesample,deltasample)], xm[c(tesample,deltasample),],
                     units=units_3,layers=layer_3,epochs=epochs_3,batch_size=batch_size, l1= l1_3 )
      eymx1_w=eymx1$w

      eymx1_loss = eymx1$ val_loss
      eymx1_loss_min = min(eymx1$val_loss_all)
      eymx1_epoch = which.min(eymx1$val_loss_all)

      # predict E(Y|M,X,D=1) in test data
      eymx1te= eymx1$y_pred[1:length(tesample)]
      # predict E(Y|M,X,D=1) in delta sample
      eymx1trte= eymx1$y_pred[-(1:length(tesample))]

    }

    # 4. fit E[E(Y|M,X,D=1)|D=0,X] in delta sample
    dtrte=d[deltasample]; xtrte=x[deltasample,]
    # predict E[E(Y|M,X,D=1)|D=0,X] in the test data
    regweymx1 = nn_reg(eymx1trte[dtrte==0],xtrte[dtrte==0,],eymx1te, x[tesample,],
                       units=units_4,layers=layer_4,epochs=epochs_4,batch_size=batch_size, l1= l1_4)
    regweymx1_w=regweymx1$w
    regweymx1te = regweymx1$y_pred

    regweymx1_loss = regweymx1$val_loss
    regweymx1_loss_min = min(regweymx1$val_loss_all)
    regweymx1_epoch = which.min(regweymx1$val_loss_all)

    #  fit E(Y|X,D=1) in total of training data with D=1 by running Y~X
    if (ybin!=1){
      # 5. predict E(Y|X,D=1) in the test data
      eyx1 =nn_reg(y[trsample[d[trsample]==1]],x[trsample[d[trsample]==1],],y[tesample,], x[tesample,],
                   units=units_5,layers=layer_5,
                   epochs=epochs_5,batch_size=batch_size, l1= l1_5)
      eyx1_w=eyx1$w
      eyx1te = eyx1$y_pred

      eyx1_loss = eyx1$val_loss
      eyx1_loss_min = min(eyx1$val_loss_all)
      eyx1_epoch = which.min(eyx1$val_loss_all)


      # 6. fit E(Y|M,X,D=0) in first training data
      eymx0=nn_reg(y[musample[d[musample]==0]],xm[musample[d[musample]==0],],y[c(tesample,deltasample),], xm[c(tesample,deltasample),],
                   units=units_6,layers=layer_6,epochs=epochs_6,batch_size=batch_size, l1= l1_6)
      eymx0_w=eymx0$w

      eymx0_loss = eymx0$val_loss
      eymx0_loss_min = min(eymx0$val_loss_all)
      eymx0_epoch = which.min(eymx0$val_loss_all)

      # plot(eymx0_t[c(tesample,deltasample) ],eymx0$y_pred);abline(a=0,b=1)

      # predict E(Y|M,X,D=0) in test data
      eymx0te=eymx0$y_pred[1:length(tesample)]
      # predict E(Y|M,X,D=0) in delta sample
      eymx0trte=eymx0$y_pred[-(1:length(tesample))]


    }

    # 7. fit E[E(Y|M,X,D=0)|D=1,X] in delta sample
    # predict E[E(Y|M,X,D=0)|D=1, X] in the test data
    regweymx0 = nn_reg(eymx0trte[dtrte==1],xtrte[dtrte==1,],eymx0te, x[tesample,], units=units_7,layers=layer_7,
                       epochs=epochs_7,batch_size=batch_size, l1= l1_7)
    regweymx0_w=regweymx0$w
    regweymx0te = regweymx0$y_pred

    regweymx0_loss = regweymx0$val_loss
    regweymx0_loss_min = min(regweymx0$val_loss_all)
    regweymx0_epoch = which.min(regweymx0$val_loss_all)


    if (ybin!=1){
      #  fit E(Y|X,D=0) in total of training data with D=0 by running Y~X
      # 8. predict E(Y|X,D=0) in the test data
      eyx0=nn_reg(y[trsample[d[trsample]==0]],x[trsample[d[trsample]==0],], y[tesample,], x[tesample,], units=units_8,layers=layer_8,
                  epochs=epochs_8,batch_size=batch_size, l1= l1_8)
      eyx0_w=eyx0$w
      eyx0te = eyx0$y_pred

      eyx0_loss = eyx0$val_loss
      eyx0_loss_min = min(eyx0$val_loss_all)
      eyx0_epoch = which.min(eyx0$val_loss_all)

    }


    # select observations satisfying trimming restriction
    sel= 1*((((1-pmxte)*pxte)>=trim) & ((1-pxte)>=trim)  & (pxte>=trim) &  (((pmxte*(1-pxte)))>=trim)   )
    # predict E(Y0,M(1)) in the test data
    temp=((1-dte)*pmxte/((1-pmxte)*pxte)*(yte-eymx0te)+dte/pxte*(eymx0te-regweymx0te)+regweymx0te)
    y0m1=c(y0m1,temp[sel==1])
    # predict E(Y0,M(0)) in the test data
    temp=(eyx0te + (1-dte)*(yte-eyx0te)/(1-pxte))
    y0m0=c(y0m0,temp[sel==1])
    # predict E(Y1,M(0)) in the test data
    temp=(dte*(1-pmxte)/(pmxte*(1-pxte))*(yte-eymx1te)+(1-dte)/(1-pxte)*(eymx1te-regweymx1te)+regweymx1te)
    y1m0=c(y1m0,temp[sel==1])
    # predict E(Y1,M(1)) in the test data
    temp=(eyx1te + dte*(yte-eyx1te)/pxte)
    y1m1=c(y1m1,temp[sel==1])
    # collect selection dummies
    selall=c(selall,sel)

    val_loss_i = rbind(pmx_loss, px_loss,
                       eymx1_loss, regweymx1_loss, eyx1_loss,
                       eymx0_loss, regweymx0_loss, eyx0_loss)


    val_loss_min_i = rbind(pmx_loss_min, px_loss_min,
                           eymx1_loss_min, regweymx1_loss_min, eyx1_loss_min,
                           eymx0_loss_min, regweymx0_loss_min, eyx0_loss_min)

    val_epoch_i = rbind(pmx_epoch, px_epoch,
                        eymx1_epoch, regweymx1_epoch, eyx1_epoch,
                        eymx0_epoch, regweymx0_epoch, eyx0_epoch)


    val_loss = cbind(val_loss, val_loss_i)
    val_loss_min = cbind(val_loss_min_i , val_loss_min)
    val_epoch = cbind( val_epoch_i , val_epoch)

  }

  # average over the crossfitting steps
  my1m1=mean(y1m1); my0m1=mean(y0m1); my1m0=mean(y1m0); my0m0=mean(y0m0)
  # compute effects
  tot=my1m1-my0m0; dir1=my1m1-my0m1; dir0=my1m0-my0m0; indir1=my1m1-my1m0; indir0=my0m1-my0m0;
  #compute variances
  vtot=mean((y1m1-y0m0-tot)^2); vdir1=mean((y1m1-y0m1-dir1)^2); vdir0=mean((y1m0-y0m0-dir0)^2);
  vindir1=mean((y1m1-y1m0-indir1)^2); vindir0=mean((y0m1-y0m0-indir0)^2); vcontrol=mean((y0m0-my0m0)^2)
  # report effects, mean of Y(0,M(0)), variances, number of non-trimmed observations
  list( "ATE"= c(tot, dir1, dir0, indir1, indir0, my0m0, vtot, vdir1, vdir0, vindir1, vindir0, vcontrol, sum(selall)),
        "val_loss"=val_loss,
        "val_loss_min"=val_loss_min,
        "val_epoch"=val_epoch)

}


























