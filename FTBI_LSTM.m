clc
clear all
tic
W=[13055; 13563; 13867; 14696; 15460; 15311; 15603; 15861; 16807; 16919; 16388; 15433; 15497; 15145; 15163; 15984; 16859; ]; % 18150; 18970; 19328; 19337; 18876
zt= (W-min(W))./ (max(W)-min(W));
l=12;
d=[];
d = pdist2(zt,zt,'euclidean');
d1=triu(d,1);
cn=(size(zt,1)*(size(zt,1)-1))/2;
mu=(sum(sum(d1)))/(cn);
for i=1:size(zt,1)
   for j=1:size(zt,1)
     tong(i,j)=(d(i,j)-mu).^2;
    end
end
tt=triu(tong,1);
sigma=((sum(sum(tt)))/cn)^0.5;
f=[];
alp=ones(size(zt,1),size(zt,1));
for i=1:size(zt,1)
    for j=1:size(zt,1)
        if d(i,j)>mu*alp(i,j)
             f(i,j)=0;
        else
             
              f(i,j)=exp(-d(i,j)/(sigma/l)); 
        end
    end
end
f;
for i=1:size(zt,1)
    tu=0;
    mau=0;
    for j=1:size(zt,1)
        tu=tu+zt(j,:)*f(i,j);
        mau=mau+f(i,j);
    end
  ztmoi(i,:)=tu./mau;
end
ztmoi;
iter=0;
Zluu=[];
exilanh=0.001
while max(max(abs(ztmoi-zt)))>exilanh 
zt=ztmoi;
iter=iter+1    
d=[];
d = pdist2(zt,zt,'euclidean');
d1=triu(d,1);
cn=(size(zt,1)*(size(zt,1)-1))/2;
mu=(sum(sum(d1)))/(cn);
tong=[];
for i=1:size(zt,1)
    for j=1:size(zt,1)
      tong(i,j)=(d(i,j)-mu).^2;
    end
end
tt=triu(tong,1);
sigma=((sum(sum(tt)))/cn)^0.5;
alp=alpha1(alp,f);
f=[];
for i=1:size(zt,1)
for j=1:size(zt,1)
     if d(i,j)>mu*alp(i,j)
       f(i,j)=0;  
     else
        f(i,j)=exp(-d(i,j)/(sigma/l)); 
    end
end
end
f;
for i=1:size(zt,1)
    tu=0;
    mau=0;
    for j=1:size(zt,1)
        tu=tu+zt(j,:)*f(i,j);
        mau=mau+f(i,j);
    end
  ztmoi(i,:)=tu/mau;
end
ztmoi;
end
[ztmoi,center]=kmeans(ztmoi,5)
dist = distfcm(center, zt); 
tmp = dist.^(-2/(2-1));
U_new =tmp./(ones(5, 1)*sum(tmp))
center=center';
forecasting =((center*U_new))';
MAPE = mean((abs(forecasting-W)./W)*100)
figure;
plot(1:17,zt,'-o')
hold on
plot(1:17,forecasting,'-o')
xlabel('Month')
ylabel('Value')
legend('Actual value','Forecasting')
hold off
%Bi-LSTM model
numTimeStepsTrain = 17;
dataTrain = zt(1:numTimeStepsTrain);
dataTest = zt(numTimeStepsTrain+1:end);
XTrain = dataTrain(1:end-1);
YTrain = dataTrain(2:end);
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 1000;
layers = [
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs', 1000, ...
    'MiniBatchSize', 50, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');
net = trainNetwork(XTrain,YTrain,layers,options);
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));
numTimeStepsTest = 5;
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end
YPred = YPred';
YTrain=YTrain;
dubaomoi=[YTrain,YPred]
YPred=YPred*(max(W)-min(W))+min(W);
dataTest=dataTest*(max(W)-min(W))+min(W);
DB=YPred;
thucte=dataTest;
SMAPE=(2/length(DB))*sum((abs(DB-thucte)/(abs(DB+thucte)))*100)
