clear;clc;close all;
load('../DataFiles/data.mat')
addpath('./functions');

%This is the tpye of model to test - currently available options: 'OLS',
%'Lasso', 'wt_percentDiff', and 'wt_local'
modelType = 'wt_local';
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
numKfold = 10;      % LOOCV: k=m
featuresIn = [1,2,3]; % features forced to be included in fit (Hierarchical principle)

% %Perform feature selection and requires that all of the original features are included
opts = statset('display','iter');
switch modelType
    case 'OLS'
        model = @OLSfit;
    case 'Lasso'
        model = @Lassofit;
    case 'wt_percentDiff'
        model = @wt_percentDiff_fit;
    case 'wt_local'
        model = @wt_local_fit;
    otherwise
        warning('Unexpected model type!')
end

[fs1,history1] = sequentialfs(model,X,y,'cv',numKfold,...
    'keepin',featuresIn,'nfeatures',nAll,'options',opts);
sse1 = min(history1.Crit); % sum of squared error from the best fit

%Perform feature selection with no requirments
[fs2,history2] = sequentialfs(model,X,y,'cv',numKfold,...
    'nfeatures',nAll,'options',opts);
sse2 = min(history2.Crit); % sum of squared error from the best fit

% Compute the mean squared error predicted from physics and model (note: for the model it should be the same as above)
[minCV, indMinCV] = min(history1.Crit);
Xfeatures = X(:,history1.In(indMinCV,:)); %Extracting only features we want

% %% Use crossval command instead to find MSE of linear regression model
% % use all suggested features with linear regression + cv
% val_model = crossval(model,Xfeatures,y,'leaveout',1);
% MSE_model = mean(val_model);
%
% % use just volume estimate with lin reg + cv
% Vol = X(:,4); % volume feature
% val_phys = crossval(@PhysMdl,Vol,y,'leaveout',1);
% MSE_phys = mean(val_phys);


%% Training vs test error for different dataset sizes for both physics and OLS prediction
%%Here I am only using features 1 through 3, 7, and 4. These were selected
%%based on the feature selection above
%%(edited): after collecting more data features 1 through 5,9,and 10 gave
%%the lowest cv error

sampleSizesTested = 20:10:length(y);

count =1;
for numTrys = 1:1000 %Does splitting multiple times because initial error is sensitive to
    numTrys
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
        switch modelType
            case 'OLS'
                SSE_test =  OLSfit(Xtrain,ytrain,Xtest,ytest);
                SSE_train = OLSfit(Xtrain,ytrain,Xtrain,ytrain);
                %mdl = fitlm(Xtrain,ytrain,'linear');
                %ypredTest = predict(mdl,Xtest);
                %ypredTrain = predict(mdl,Xtrain);
            case 'Lasso'
                SSE_test = Lassofit(Xtrain,ytrain,Xtest,ytest);
                SSE_train = Lassofit(Xtrain,ytrain,Xtrain,ytrain);
                %[B,FitInfo] = lasso(Xtrain,ytrain,'CV',10);      % train and create a linear regression model
                %ypredTest = Xtest*B(:,FitInfo.Index1SE)+FitInfo.Intercept(FitInfo.Index1SE);
                %ypredTrain = Xtrain*B(:,FitInfo.Index1SE)+FitInfo.Intercept(FitInfo.Index1SE);
            case 'wt_percentDiff'
                SSE_test = wt_percentDiff_fit(Xtrain,ytrain,Xtest,ytest);
                SSE_train = wt_percentDiff_fit(Xtrain,ytrain,Xtrain,ytrain);
                %mdl = fitlm(Xtrain,ytrain,'linear','weights',ytrain);
                %ypredTest = predict(mdl,Xtest);
                %ypredTrain = predict(mdl,Xtrain);    
            case 'wt_local'
                SSE_test = wt_local_fit(Xtrain,ytrain,Xtest,ytest);
                SSE_train = wt_local_fit(Xtrain,ytrain,Xtrain,ytrain);
            otherwise
                warning('Unexpected model type!')
        end
        
        %         if RoundPrediction
        %             ypredTest  = round(4*ypredTest)/4;
        %             ypredTrain = round(4*ypredTrain)/4;
        %         end
%         trainError(count,numTrys) = mean((ytrain-ypredTrain).^2);
%         testError(count,numTrys)  = mean((ytest-ypredTest).^2);
        [mtrain,ntrain] = size(Xtrain);
        [mtest,ntest] = size(Xtest);

        trainError(count,numTrys) = SSE_train/mtrain;
        testError(count,numTrys)  = SSE_test/mtest;
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
%print('./Figures/eps/featureSelectionLasso','-depsc')
%print('./Figures/jpegs/featureSelectionLasso','-djpeg','-r600')


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
%print('./Figures/eps/dataSampleSizeLasso','-depsc')
%print('./Figures/jpegs/dataSampleSizeLasso','-djpeg','-r600')
