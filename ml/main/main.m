clc
clear all
close all

trainingUsage=0.9;
padding='n';
neighborhoodSize=2;
epochs=175;
k = 5;

% HPC Config
pc=parcluster('local');
parpool(pc,48) % max is 48 on the cluster, verified by admin

mkdir('savedData')
mkdir('plots')

JV=importdata('N17_6cleanJV.mat');
JV(:,2:2:end) = JV(:,2:2:end)*(-1.0)*10^6;

mat=importdata('N17_6deltaV2024.mat');

[allJV,allMat]=neighborhood(JV,mat,neighborhoodSize,padding);

rand_i = randperm(length(allJV)); 
train_stopIndex = round(length(allJV)*trainingUsage);

% split into test set, train set 
testSetJV = allJV(train_stopIndex+1:end);
testSetMat = allMat(train_stopIndex+1:end);

% seperated set for k-fold, training, and model selection
trainSetJV = allJV(1:train_stopIndex);
trainSetMat = allMat(1:train_stopIndex);

%% k-fold cross validation
lr=linspace(1e-5,1e-3,50); % grid search for best learning rate
folds = 1:k;
cvIndices = crossvalind('Kfold',length(trainSetMat),k);
% cvError = zeros(length(lr),k);

gridMatrix = combvec(lr,folds);
grid_error = zeros(1,length(gridMatrix));
best_i = zeros(1,length(gridMatrix));

parfor i = 1:length(gridMatrix)
    
    learning_rate = gridMatrix(1,i);

    train_i = (cvIndices ~= gridMatrix(2,i));
    valid_i = (cvIndices == gridMatrix(2,i));

    tJV = trainSetJV(train_i);
    tMat = trainSetMat(train_i);
    vJV = trainSetJV(valid_i);
    vMat = trainSetMat(valid_i);

    max_mat=max(tMat);
    min_mat=min(tMat);
    tMat = (tMat-min_mat)./(max_mat-min_mat);
    vMat = (vMat-min_mat)./(max_mat-min_mat);

    [trainingLoss,validationLoss,~] = smallNetFunct(learning_rate,epochs,0,tJV,tMat,vJV,vMat);

    [grid_error(i),best_i(i)]=min(validationLoss);

    formattedLR = sprintf('%.2e', learning_rate);
    currentFold = int2str(gridMatrix(2,i));
    figure();
    plot(trainingLoss,'b')
    hold on
    plot(validationLoss,'r')
    xlabel('Epochs')
    axis square
    grid on
    ylabel('Error (MAPE)')
    legend('Training','Validation')
    title("k: "+currentFold+" lr: "+formattedLR+" Training Curve"+"; " + ...
        "Best: "+int2str(best_i(i))+"; T.Acc: "+num2str(trainingLoss(best_i(i)))+"; V.Acc: "+num2str(validationLoss(best_i(i))))

    figureName = "gridPos"+int2str(i)+".png";
    saveas(gcf,fullfile('plots',figureName));
end

%% Final training with best hyperparameters

% get best hyperparameter
cvError = reshape(grid_error,length(lr),k);

best_i = reshape(best_i,length(lr),k);
[~,linIndex] = max(best_i(:));
[row_i, col_i] = ind2sub(size(best_i),linIndex);
earlyStopEpoch = best_i(row_i,col_i);

[~,bestIndex] = min(mean(cvError,2));

disp('k-fold performance')
disp(cvError)
disp(" ")
disp("best lr: "+lr(bestIndex))
disp("cv final training epochs: "+earlyStopEpoch)

max_matF=max(trainSetMat);
min_matF=min(trainSetMat);
finaltMat = (trainSetMat-min_matF)./(max_matF-min_matF);

save(fullfile('savedData','max_matF.mat'),'max_matF')
save(fullfile('savedData','min_matF.mat'),'min_matF')

[trainingLoss,~,encoderNet] = smallNetFunct(lr(bestIndex),epochs,earlyStopEpoch,trainSetJV,finaltMat);

disp("Final Training Pass Error: "+min(trainingLoss))
save(fullfile('savedData','trainingLoss.mat'),'trainingLoss')
save(fullfile('savedData','encoderNet.mat'),'encoderNet')

%% Final test (performance evaluation) using the test set that was left out

myNet = encoderNet{1};
testSetMat = (testSetMat - min_matF)./(max_matF-min_matF);
onePush_testJV=zeros(28,9,2,length(testSetJV));
for k=1:length(testSetJV)
    onePush_testJV(:,:,:,k)=testSetJV{k};
end
onePush_testJV=dlarray(onePush_testJV,'SSCB');

[mapeError, mseError, maeError, rmsleError, rsError] = allLoss(myNet,onePush_testJV,testSetMat);

disp("mapeError: "+mapeError)
disp("mseError: "+mseError)
disp("maeError: "+maeError)
disp("rmsleError: "+rmsleError)
disp("rsError: "+rsError)

%% Finally, making a prediction map
onePush_allJV=zeros(28,9,2,length(allJV));
for k=1:length(allJV)
    onePush_allJV(:,:,:,k)=allJV{k};
end
onePush_allJV=dlarray(onePush_allJV,'SSCB');

allPred=extractdata(predict(myNet,onePush_allJV));
save(fullfile('savedData','allPred.mat'),'allPred')

visualizeMap(allPred,'',sqrt(length(allPred)),0.025)
saveas(gcf,fullfile('plots','paramMap.png'));

% HPC Config
poolobj = gcp('nocreate');
delete(poolobj);




