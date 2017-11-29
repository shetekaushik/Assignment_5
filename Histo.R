data<-read.csv(file="Housing.csv",head=TRUE,sep=",")

varience<-function(data){
  sqrt(sd(data))
}

range1<-function(data){
  max(data)-min(data)
}

get_mode<-function(data){
  uniq<-unique(data)
  uniq[which.max(tabulate(match(data,uniq)))]
}

z_score<-function(data,user){
  (user-min(data))/sd(data)
}

min_max<-function(data,user,new_min,new_max){
  ((user-min(data))/(max(data)-min(data)))*(new_max-new_min)+new_min
}

decimal<-function(user,scale1){
  (user/scale1)
}


HistGram<-function(data,breaks,X_min_lim,X_max_lim,Y_min_lim,X_max_lim){

hist.default(data,breaks,xlim = c(X_min_lim,X_max_lim),ylim = c(Y_min_lim,Y_max_lim))
}

