#Clean the environment

rm(list=ls())

##Set the working directory

setwd("D:/Amol_Data/Edwisor/Assignments/Project_2")
getwd()

##Load the required libraries

libraries = c("ggplot2","plyr","dplyr","rpart","gbm","DMwR","randomForest","usdm","corrgram","DataCombine")
lapply(X=libraries,require,character.only=TRUE)

#Load the dataset
data_day = read.csv(file="day.csv",header = T,sep=",",na.strings = c(" ","","NA"))


#######################Exploraty Data Analysis###############################################################

head(data_day)

dim(data_day)
str(data_day)

##Droping data which is not essential
#instant- index number
#dteday - all the required values are present
#casual,registered - cnt variable is  the sum of casual and registered variable

data_day <- subset(data_day,select = -c(instant,dteday,casual,registered))


data_day$season = as.factor(data_day$season)
data_day$yr =as.factor(data_day$yr)
data_day$mnth = as.factor(data_day$mnth)
data_day$weekday =as.factor(data_day$weekday)
data_day$workingday = as.factor(data_day$workingday)
data_day$holiday = as.factor(data_day$holiday)


###################################Univarent Analysis####################################

data_vis <- data_day

data_vis$season = factor( x = data_vis$season,level = c(1,2,3,4),labels = c("Spring","Summer","Fall","Winter"))
data_vis$yr = factor(x=data_vis$yr,levels = c(0,1),labels = c("2011","2012"))
data_vis$holiday = factor(x=data_vis$holiday,levels = c(0,1),labels = c("Working","Holiday"))
data_vis$weathersit =factor(x=data_vis$weathersit,levels = c(1,2,3,4),labels = c("Clear","Cloudy/Mist","Rain/Snow?Fog","Heavy Rain/Snow/Fog"))


bar1 = ggplot(data = data_vis,aes(x = season)) +geom_bar() + ggtitle("Season")
bar2 = ggplot(data = data_vis,aes(x=workingday)) + geom_bar() + ggtitle("working dayr")
bar3 = ggplot(data = data_vis,aes(x=holiday)) + geom_bar() + ggtitle("Holiday")
bar4 = ggplot(data = data_vis,aes(x=weathersit)) + geom_bar() + ggtitle("weather")

gridExtra::grid.arrange(bar1,bar2,bar3,bar4,ncol=2)

##################################### Bivarient Analysis########################################

sct1 = ggplot(data=data_vis,aes(x=temp,y=cnt)) + ggtitle("Temp") +geom_point() + xlab("Temperature") +ylab("Bike count")
sct2 = ggplot(data=data_vis,aes(x=atemp,y=cnt)) + ggtitle("aTemp") +geom_point() + xlab("Temperature") +ylab("Bike count")
sct3 = ggplot(data=data_vis,aes(x=hum,y=cnt)) + ggtitle("Humidity") +geom_point() + xlab("Humidity") +ylab("Bike count")
sct4 = ggplot(data=data_vis,aes(x=windspeed,y=cnt)) + ggtitle("Windspeed") +geom_point() + xlab("windspeed") +ylab("Bike count")
gridExtra::grid.arrange(sct1,sct2,sct3,sct4,ncol=2)


####################################BOx plot  for outlier analysis################################
cnames = colnames(data_day[,c("temp","atemp","hum","windspeed")])
cnames
 for( i in 1:length(cnames))
 {
   assign(paste0("gn",i),ggplot(aes_string(x=cnames[i],y='cnt'), data=data_day) +
            stat_boxplot(geom = "errorbar",width =0.5)+
            geom_boxplot(outlier.colour = "red",fill="grey",outlier.shape =18,
                         outlier.size=1,notch=FALSE)+
          theme(legend.position = "bottom")+
            labs(x=cnames[i])+
            ggtitle(paste("Box plot for",cnames[i])))
 }

gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=4)

cor(data_day$temp,data_day$cnt)
cor(data_day$hum,data_day$cnt)

##Remove outlier in windspeed

val = data_day$windspeed[data_day$windspeed %in% boxplot.stats(data_day$windspeed)$out]
val
data_day = data_day[which(!data_day$windspeed %in% val),]


val = data_day$hum[data_day$hum %in% boxplot.stats(data_day$hum)$out]
val
data_day = data_day[which(!data_day$hum %in% val),]


cnames = colnames(data_day[,c("hum","windspeed")])
cnames
for( i in 1:length(cnames))
{
  assign(paste0("gn",i),ggplot(aes_string(x=cnames[i],y='cnt'), data=data_day) +
           stat_boxplot(geom = "errorbar",width =0.5)+
           geom_boxplot(outlier.colour = "red",fill="grey",outlier.shape =18,
                        outlier.size=1,notch=FALSE)+
           theme(legend.position = "bottom")+
           labs(x=cnames[i])+
           ggtitle(paste("Box plot for",cnames[i])))
}

gridExtra::grid.arrange(gn1,gn2,ncol=2)

#####################################Missing Value Analysis#################################
missing_values = sapply(data_day,function(x){sum(is.na(x))})
print(missing_values)

## No missing Values

########################################### Feature scaling##################################

##In dataset numeric continous variables "temp","atemp","hum",and "windspeed" are in normalized form
## No need to feature scaling

########################################## Feature selection ################################

df_vif = data_day[,c("temp","atemp","hum","windspeed")]
vifcor(df_vif)


corrgram(data_day,order =F,upper.panel = panel.pie, text.panel = panel.txt,main="Correlation Plot")

names(data_day)
data_day <- subset(data_day,select = c("season","yr","mnth","weekday","temp","hum","windspeed","cnt"))

rmExcept(keepers = "data_day")


##Split data into train and test
set.seed(123)

train_index = sample(1:nrow(data_day),0.8*nrow(data_day))
train = data_day[train_index,]
test = data_day[-train_index,]



############################################   Linear Regression #############################

#Train
LR_model = lm(formula = cnt~.,data = train)

##summary

summary(LR_model)

LR_model_prediction = predict(LR_model,test[,-8])

df = data.frame("actual_value" =test[,8],"LR_model_predict"= LR_model_prediction)
head(df)


regr.eval(trues = test[,8],preds = LR_model_prediction,stats = c("mae","mse","rmse","mape"))

## mae          mse         rmse         mape 
## 6.422606e+02 7.366696e+05 8.582946e+02 2.171891e-01 


######################################Decision tree ################################

set.seed(12)
DT_model = rpart(cnt~.,data = train,method = "anova")

DT_model_predict = predict(DT_model,test[,-8])

df = data.frame(df,DT_model_predict)
head(df)
par(cex = 0.8)
plot(DT_model)
text(DT_model)


regr.eval(trues = test[,8],preds = DT_model_predict,stats = c("mae","mse","rmse","mape"))

## mae          mse         rmse         mape 
## 7.329756e+02 9.471663e+05 9.732247e+02 2.831340e-01 



##################################### Random Forest ##################################

RF_model = randomForest(cnt~.,data=train,ntree=500,nodesize=8,importance=TRUE)
RF_model_predict = predict(RF_model,test[,-8])
df = cbind(df,RF_model_predict)
head(df)


regr.eval(trues = test[,8],preds = RF_model_predict,stat = c("mae","mse","rmse","mape"))

## mae          mse         rmse         mape 
## 6.033788e+02 6.486310e+05 8.053763e+02 2.436907e-01 

############  Result   ##################
write.csv(df,"Test_result_R.csv",row.names = FALSE)
## From Above three model's RMSE metrics we can say that Random Forest works better than Decision Tree and Linear Regression