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
Name:-Kaushik Shete
