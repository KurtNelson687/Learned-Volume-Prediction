clear;clc;close all;
load('../DataFiles/data.mat')
addpath('./functions');

%This is the tpye of model to test - currently available options: 'OLS',
%'Lasso', 'Ridge', 'wt_percentDiff', 'Physical', 'wt_local', and 'KNN'
modelType = 'wt_local'; %Select model type
RoundPrediction = 0; % choose to round the output of linear regression
testDataSize = 0; % Include data size test or not
numTrials = 100; % number of trails for each case
tablespoon_to_cup = 16; % tablespoon to cups conversion 
numKfold = 10;      % number of folds for cross validation

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
featuresIn = [1,2,3]; % features forced to be included in fit (Hierarchical principle)

% %Perform feature selection and requires that all of the original features are included
opts = statset('display','iter');
switch modelType
    case 'OLS'
        model = @OLSfit;
    case 'Lasso'
        model = @Lassofit;
    case 'Ridge'
        model = @Ridgefit;
    case 'wt_percentDiff'
        model = @wt_percentDiff_fit;
    case 'wt_local'
        model = @wt_local_fit;
    case 'KNN'
        model = @KNNfit_reg6;
    case 'Physical'
        model = @PhysMdl;
    otherwise
        warning('Unexpected model type!')
end

if ~strcmp(modelType,'Physical')
    for trail = 1:numTrials
        trail
        [fs1,history1] = sequentialfs(model,X,y,'cv',numKfold,...
            'keepin',featuresIn,'nfeatures',nAll,'options',opts);
        sse1 = min(history1.Crit); % sum of squared error from the best fit
        CVerror(trail,:) = history1.Crit;
        
        %Perform feature selection with no requirments
        %[fs2,history2] = sequentialfs(model,X,y,'cv',numKfold,...
        %    'nfeatures',nAll,'options',opts);
        % sse2 = min(history2.Crit); % sum of squared error from the best fit
        % Compute the mean squared error predicted from physics and model (note: for the model it should be the same as above)
    end
else
    for trail = 1:numTrials
        CVerror(trail) = mean(crossval(model,X(:,4),y,'kfold',numKfold)/numKfold);
    end
end

%Compute mean and standard errors for the cross-validation error;
if ~strcmp(modelType,'Physical')
    MSE_mean = mean(CVerror);
    [minMSE, indMin] = min( MSE_mean)
    std_MSE = std(CVerror(:,indMin));
    SE_MSE = std_MSE/sqrt(numTrials)
    
    RMSEerror = sqrt(CVerror);
    RMSE_mean = mean(RMSEerror);
    minRMSE =  RMSE_mean(indMin)
    std_RMSE = std(RMSEerror(:,indMin));
    SE_RMSE =1.95*std_RMSE/sqrt(numTrials)
    
    Xfeatures = X(:,history1.In(indMin,:));
else
    [minMSE, indMin] = min(mean(CVerror))
    std_MSE = std(CVerror);
    SE_MSE = 1.95*std_MSE/sqrt(numTrials)
    
    RMSEerror = sqrt(CVerror);
    RMSE_mean = mean(RMSEerror);
    minRMSE = RMSE_mean
    std_RMSE = std(RMSEerror);
    SE_RMSE = std_RMSE/sqrt(numTrials)
    
    Xfeatures = X(:,4);
    SE = std(CVerror)/sqrt(numTrials);
end


%% Training vs test error for different dataset sizes for both physics and OLS prediction
%%Here I am only using features 1 through 3, 7, and 4. These were selected
%%based on the feature selection above
%%(edited): after collecting more data features 1 through 5,9,and 10 gave
%%the lowest cv error

if testDataSize
    sampleSizesTested = 20:10:length(y);
    count =1;
    for numTrys = 1:2000 %Does splitting multiple times because initial error is sensitive to
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
                case 'Lasso'
                    SSE_test = Lassofit(Xtrain,ytrain,Xtest,ytest);
                    SSE_train = Lassofit(Xtrain,ytrain,Xtrain,ytrain);
                case 'Ridge'
                    SSE_test = Ridgefit(Xtrain,ytrain,Xtest,ytest);
                    SSE_train = Ridgefit(Xtrain,ytrain,Xtrain,ytrain);
                case 'wt_percentDiff'
                    SSE_test = wt_percentDiff_fit(Xtrain,ytrain,Xtest,ytest);
                    SSE_train = wt_percentDiff_fit(Xtrain,ytrain,Xtrain,ytrain);
                case 'wt_local'
                    SSE_test = wt_local_fit(Xtrain,ytrain,Xtest,ytest);
                    SSE_train = wt_local_fit(Xtrain,ytrain,Xtrain,ytrain);
                case 'KNN'
                    SSE_test  = KNNfit_reg3(Xtrain,ytrain,Xtest,ytest);
                    SSE_train = KNNfit_reg3(Xtrain,ytrain,Xtrain,ytrain);
                case 'Physical'
                    SSE_test  = PhysMdl(Xtrain,ytrain,Xtest,ytest);
                    SSE_train = PhysMdl(Xtrain,ytrain,Xtrain,ytrain);
                otherwise
                    warning('Unexpected model type!')
            end
            
            [mtrain,ntrain] = size(Xtrain);
            [mtest,ntest] = size(Xtest);
            
            trainError(count,numTrys) = SSE_train/mtrain;
            testError(count,numTrys)  = SSE_test/mtest;
            count = count+1;
        end
    end
    testError = mean(testError,2);
    trainError = mean(trainError,2);
end

%% Plot stuff
if ~strcmp(modelType,'Physical')
    fig1 = figure;%('visible', 'off');
    fig1.PaperUnits = 'centimeters';
    fig1.PaperPosition = [0 0 8 4];
    set(gca,'box','on')
    plot(length(featuresIn):nAll,MSE_mean,'linewidth',1)
    ylab = ylabel('$CV_{\mathrm{MSE}}$');
    set(ylab,'interpreter','Latex','FontSize',10)
    xlab = xlabel('Number of features');
    set(xlab,'interpreter','Latex','FontSize',10)
    set(gca,'FontSize',8)
    %print('./Figures/eps/featureSelectionWLS','-depsc')
    %print('./Figures/jpegs/featureSelectionWLS','-djpeg','-r600')
end

if testDataSize
    fig2 = figure;%('visible', 'off');
    fig2.PaperUnits = 'centimeters';
    fig2.PaperPosition = [0 0 8 4];
    set(gca,'box','on')
    plot(sampleSizesTested,trainError,'r','linewidth',1)
    hold
    plot(sampleSizesTested,testError,'k','linewidth',1)
    ylab = ylabel('MSE');
    set(ylab,'interpreter','Latex','FontSize',10)
    xlab = xlabel('Data sample size');
    set(xlab,'interpreter','Latex','FontSize',10)
    set(gca,'FontSize',8,'xlim',[30 200])
    leg = legend('training error', 'development error');
    set(leg,'interpreter','Latex','FontSize',8)
    print('./Figures/eps/dataSampleSizeWLS','-depsc')
    print('./Figures/jpegs/dataSampleSizeWLS','-djpeg','-r600')
    
    if strcmp(modelType,'Lasso')
        %Normalize variables
        [Xfeatures,mu, s] = normalizeVars(Xfeatures);
        [B,FitInfo] = lasso(Xfeatures,y,'CV',10);
        
        fig3 = figure;%('visible', 'off');
        lassoPlot(B,FitInfo,'PlotType','CV');
        print('./Figures/eps/LassoCV','-depsc')
        print('./Figures/jpegs/LassoCV','-djpeg','-r600')
        
        fig4 = figure;%('visible', 'off');
        lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log');
        print('./Figures/eps/LassoBetas','-depsc')
        print('./Figures/jpegs/LassoBetas','-djpeg','-r600')
    end
end