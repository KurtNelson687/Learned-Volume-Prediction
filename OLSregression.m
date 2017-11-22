clear;clc;close all;
load('FitData_usable.mat')

%% create possible feature sets
% Rename and remove intercept feature
X=Xgood(:,2:end);
y=ygood;
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

%Perform feature selection and require that all of the original features are included
opts = statset('display','iter');
[fs1,history1] = sequentialfs(@OLSfit,X,y,'cv',numKfold,...
    'keepin',featuresIn,'nfeatures',nAll,'options',opts);
sse1 = min(history1.Crit)/1; % sum of squared error from the best fit (sse=mse, since 1 test point)
% Sequential feature selection: performs cross-val on numKfolds, forces
% alg. to keep in the 3 fundamental features, makes sure the alg. performs
% over all possible features (doesn't stop at a local min), and display
% information at each sequential iteration. 

%Perform feature selection with no requirments
[fs2,history2] = sequentialfs(@OLSfit,X,y,'cv',numKfold,...
    'nfeatures',nAll,'options',opts);
sse2 = min(history2.Crit); % sum of squared error from the best fit

% %% Compute the mean squared error predicted from physics and model (note: for the model it should be the same as above)
Xfeatures = X(:,[1;2;3;7;4]); %Extracting only features we want
% for i = 1:mAll
%     i
%     %For physics prediction
%     Vol = X(:,4);
%     Vol(i)=NaN;
%     Vol = Vol(~isnan(Vol));
%     
%     ytrain = y;
%     ytrain(i)=NaN;
%     ytrain = ytrain(~isnan(ytrain));
%     
%     scale = nanmean(ytrain./Vol); %Scaling for physics prediction
%     yPhs = scale*X(i,4);
%     physicsError(i) = mean((y(i)-yPhs).^2);
%     
%     %For model
%     Xtrain = Xfeatures;
%     Xtrain(i,:)=NaN;
%     Xtrain = Xtrain(~isnan(Xtrain(:,1)),:);
%    
%     
%     Xtest = Xfeatures(i,:);
%     
%      mdl = fitlm(Xtrain,ytrain,'linear');
%      ypredTest = predict(mdl,Xtest);
%      modelError(i) = mean((y(i)-ypredTest).^2);
% end
% MSE_physics = mean(physicsError);
% MSE_model = mean(modelError);

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
sampleSizesTested = 20:10:100;

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
