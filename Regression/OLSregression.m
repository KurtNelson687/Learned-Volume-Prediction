clear;clc;close all;
load('../DataFiles/data.mat')
addpath('./functions');

RoundPrediction = 0; % choose to round the output of linear regression

%% create possible feature sets
% Rename and remove intercept feature
X=X_train(:,2:end);
y=y_train;
y = y';

%Adds columns to X so that all second order terms of original features are included
X(:,5) = X(:,1).^2; %square of duration
X(:,6) = X(:,2).^2; %square of front speed
X(:,7) = X(:,3).^2; %square of area

%Adds all two-way interaction terms to X
X(:,8) = X(:,1).*X(:,2); %duration and front speed
X(:,9) = X(:,1).*X(:,3); %duration and area
X(:,10) = X(:,2).*X(:,3); %front speed and area
[mAll, nAll] = size(X);

%% Perform Feature selection using linear regression and LOOCV
% Model feature selection performing k-fold cross validation
numKfold = mAll;      % LOOCV: k=m
featuresIn = [1,2,3]; % features forced to be included in fit (Hierarchical principle)

%Perform feature selection and requires that all of the original features are included
opts = statset('display','iter');
[fs1,history1] = sequentialfs(@OLSfit,X,y,'cv',numKfold,...
    'keepin',featuresIn,'nfeatures',nAll,'options',opts);
sse1 = min(history1.Crit); % sum of squared error from the best fit  

%Perform feature selection with no requirments
[fs2,history2] = sequentialfs(@OLSfit,X,y,'cv',numKfold,...
    'nfeatures',nAll,'options',opts);
sse2 = min(history2.Crit); % sum of squared error from the best fit

% %Perform feature selection for lasso regression
% [fs3,history3] = sequentialfs(@Lassofit,X,y,'cv',numKfold,...
%     'nfeatures',nAll,'options',opts);
% sse3 = min(history3.Crit); % sum of squared error from the best fit

% Compute the mean squared error predicted from physics and model (note: for the model it should be the same as above)
Xfeatures = X(:,[1;2;3;5;9;10]); %Extracting only features we want

%% Use crossval command instead to find MSE of linear regression model
% use all suggested features with linear regression + cv
val_OLS = crossval(@OLSfit,Xfeatures,y,'leaveout',1);
MSE_OLS = mean(val_OLS);

% use just volume estimate with lin reg + cv
Vol = X(:,4); % volume feature
val_phys = crossval(@PhysMdl,Vol,y,'leaveout',1);
MSE_phys = mean(val_phys);

% use all suggested features with weighted linear regression + cv
val_wtLS = crossval(@wtLSfit,Xfeatures,y,'leaveout',1);
MSE_wtLS = mean(val_wtLS);


%% Training vs test error for different dataset sizes for both physics and OLS prediction
%%Here I am only using features 1 through 3, 7, and 4. These were selected
%%based on the feature selection above
%%(edited): after collecting more data features 1 through 5,9,and 10 gave
%%the lowest cv error

sampleSizesTested = 20:20:180;

count =1;
for numTrys = 1:1000 %Does splitting multiple times because initial error is sensitive to 
    count =1;
    for sampleSize = sampleSizesTested
        [Xsubset,idx] = datasample(Xfeatures,sampleSize,'replace',false);
        ysubset = y(idx);
        [trainInd,~,testInd] = dividerand(sampleSize,.7,0,.3);
        
        %Setup training and test data
        Xtrain = Xsubset(trainInd,:);
        ytrain = ysubset(trainInd,:);
        
        Xtest = Xsubset(testInd,:);
        ytest = ysubset(testInd,:);
        
        %fit linear model with volume interaction and squared area
        mdl = fitlm(Xtrain,ytrain,'linear');
        ypredTest = predict(mdl,Xtest);
        ypredTrain = predict(mdl,Xtrain);
        if RoundPrediction
            ypredTest  = round(4*ypredTest)/4;
            ypredTrain = round(4*ypredTrain)/4;
        end
        
        trainError(count,numTrys) = mean((ytrain-ypredTrain).^2);
        testError(count,numTrys)  = mean((ytest-ypredTest).^2);
        count = count+1;
    end
end
testError = mean(testError,2);
trainError = mean(trainError,2);

%% Plot stuff
fig1 = figure;%('visible', 'off');
fig1.PaperUnits = 'centimeters';
fig1.PaperPosition = [0 0 8 4];
set(gca,'box','on')
plot(length(featuresIn):nAll,history1.Crit,'linewidth',1)
hold
plot(1:nAll,history2.Crit,'linewidth',1)
ylab = ylabel('CV');
set(ylab,'interpreter','Latex','FontSize',8)
xlab = xlabel('Number of features');
set(xlab,'interpreter','Latex','FontSize',8)
set(gca,'FontSize',6)
leg = legend('Enforcing hierarchical principle', 'Unrestricted');
set(leg,'interpreter','Latex','FontSize',6)
% print('./Figures/eps/WriteUp/featureSelection','-depsc')
% print('./Figures/jpegs/WriteUp/featureSelection','-djpeg','-r600')


fig2 = figure;%('visible', 'off');
fig2.PaperUnits = 'centimeters';
fig2.PaperPosition = [0 0 8 4];
set(gca,'box','on')
plot(sampleSizesTested,trainError,'r','linewidth',1)
hold
plot(sampleSizesTested,testError,'k','linewidth',1)
ylab = ylabel('MSE');
set(ylab,'interpreter','Latex','FontSize',8)
xlab = xlabel('Data sample size');
set(xlab,'interpreter','Latex','FontSize',8)
set(gca,'FontSize',6)
leg = legend('training error', 'test error');
set(leg,'interpreter','Latex','FontSize',6)
% print('./Figures/eps/WriteUp/dataSampleSize','-depsc')
% print('./Figures/jpegs/WriteUp/dataSampleSize','-djpeg','-r600')