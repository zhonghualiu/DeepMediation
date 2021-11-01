# built up NN

nn_reg <- function(y,x,y_test,x_test,units=3,layers=1,epochs=500,batch_size=100,verbose=0,
                   early_stop="F",delta_stop=0,epoch_stop=50,val_prop=0,
                   report_loss="T",drop_rate=0, lr=0.001,l1=0){
  # l1 = 0
  if(is.vector(x)){x = as.matrix(x)}
  
  model <- keras_model_sequential() 
  i = 0
  while(i<layers){
    if (i==0){ n_input = ncol(x); lambda=l1; units_i=  units
    }else{ n_input = units; lambda=0; units_i=units  }
    
    #model %>% layer_batch_normalization()
    model %>% layer_dense(units = units_i, activation = 'relu', input_shape = n_input, 
                          kernel_regularizer = regularizer_l1(lambda)
                          ,kernel_initializer = initializer_glorot_uniform(1)
                          #,bias_initializer = "zero"
                          )  # hidden layers
     model %>% layer_batch_normalization()
    # if(i>0){model %>% layer_dropout(rate = drop_rate)}
    i = i+1
  }
  
  #model %>% layer_batch_normalization()
  model %>% layer_dense(units = 1
                        ,kernel_initializer = initializer_glorot_uniform(1)
                        #,bias_initializer = "zero"
                        ) # output layer
  # summary(model)
  
  model %>% compile(
    loss = "mean_squared_error",
     optimizer =  optimizer_adam(lr=lr,decay = 0), 
    # optimizer =  optimizer_sgd(lr=lr,decay = 0), 
    metrics = "mean_squared_error"
  )
  if(early_stop=="F"){epoch_stop=epochs}
  NNfit <- model %>% fit(x, y, epochs = epochs, batch_size=batch_size, verbose = verbose,
                         validation_split = val_prop, validation_data = list (x_test, y_test),
                         callbacks=list(
                           callback_early_stopping(
                             monitor = "val_loss",
                             min_delta = delta_stop,
                             patience = epoch_stop,
                             mode = "auto",
                             baseline = NULL,
                             restore_best_weights = FALSE
                           )
                           #,callback_reduce_lr_on_plateau()
                         )
  )
  w=get_weights(model)[[1]]
  w=rowMeans(abs(w))
  
  y_pred = model %>% predict(x_test)
  
  if(report_loss=="T"){
    epoch_tot = length(NNfit[["metrics"]][["loss"]])
    tr_loss = NNfit[["metrics"]][["loss"]][epoch_tot]
    val_loss = NNfit[["metrics"]][["val_loss"]][epoch_tot]
    val_loss_all = NNfit[["metrics"]][["val_loss"]]
    
    return (list("y_pred"=y_pred, NNfit,"val_loss_all"=val_loss_all,
                 "tr_loss"=tr_loss,"val_loss"=val_loss,"epoch_tot"=epoch_tot,"w"=w))
  } else{return (y_pred)}
}



nn_bin <- function(y,x,y_test,x_test,units=3,layers=1,epochs=500,batch_size=100,verbose=0,
                   early_stop="F",delta_stop=0,epoch_stop=50,val_prop=0,
                   report_loss="T",drop_rate=0, lr=0.001, l1=0){
  # l1 = 0
  if(is.vector(x)){x = as.matrix(x)}
  
  model <- keras_model_sequential() 
  i = 0
  while(i<layers){
    if (i==0){ n_input = ncol(x); lambda=l1; units_i=  units
    }else{ n_input = units; lambda=0;  units_i=units }
    
    #model %>% layer_batch_normalization()
    model %>% layer_dense(units = units_i, activation = 'relu', input_shape = n_input,
                          kernel_regularizer = regularizer_l1(lambda)
                          ,kernel_initializer = initializer_glorot_uniform(1)
                          #,bias_initializer = "zero"
                          )  # hidden layers
    model %>% layer_batch_normalization()
    if(i>0){
      model %>%layer_dropout(rate = drop_rate)
    }
    i = i+1
  }
  #model %>% layer_batch_normalization()
  model %>% layer_dense(units=1, activation="sigmoid"
                        ,kernel_initializer = initializer_glorot_uniform(1)
                        #,bias_initializer = "zero"
                        ) # output layer
  model %>% compile(
    loss = 'binary_crossentropy',
    #loss="mean_squared_error",
    optimizer =  optimizer_adam(lr=lr,decay = 0), 
   #optimizer =  optimizer_sgd(lr=lr,decay = 0), 
    metrics = "binary_crossentropy"
  )
  if(early_stop=="F"){epoch_stop=epochs}
  NNfit <- model %>% fit(x, y, epochs = epochs, batch_size=batch_size, verbose = verbose,
                         validation_split = val_prop,  validation_data = list (x_test, y_test),
                         callbacks=list(
                           callback_early_stopping(
                             monitor = "val_loss",
                             min_delta = delta_stop,
                             patience = epoch_stop,
                             mode = "auto",
                             baseline = NULL,
                             restore_best_weights = FALSE
                           )
                           #,callback_reduce_lr_on_plateau()
                         )
  )
  w=get_weights(model)[[1]]
  w=abs(w)
  w=rowMeans(w)
  #w=get_weights(model)
  
  
  y_pred = predict_proba (model, x_test)
  
  if(report_loss=="T"){
    epoch_tot = length(NNfit[["metrics"]][["loss"]])
    tr_loss = NNfit[["metrics"]][["loss"]][epoch_tot]
    val_loss = NNfit[["metrics"]][["val_loss"]][epoch_tot]
    val_loss_all = NNfit[["metrics"]][["val_loss"]]
    
    return (list("y_pred"=y_pred, NNfit,"val_loss_all"=val_loss_all,
                 "tr_loss"=tr_loss,"val_loss"=val_loss,"epoch_tot"=epoch_tot,"w"=w))
  } else{return (y_pred)}
}












































