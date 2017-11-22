clear;clc;close all;
% load('CleanedFitData.mat')
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
 
% Model feature selection performing k-fold cross validation
numKfold = mAll;
featuresIn = [1,2,3]; %features forced to be included in fit

opts = statset('display','iter');
%Perform feature selection and require that all of the original features are included
[fs1,history1] = sequentialfs(@LogisticFit_KN,X,y,'cv',numKfold,...
    'keepin',featuresIn,'nfeatures',nAll,'options',opts);

%Perform feature selection with no requirments
[fs2,history2] = sequentialfs(@LogisticFit_KN,X,y,'cv',numKfold,...
    'nfeatures',nAll,'options',opts);

fig1 = figure;%('visible', 'off');
fig1.PaperUnits = 'centimeters';
fig1.PaperPosition = [0 0 8 4];
set(gca,'box','on')
plot(length(featuresIn):nAll,history1.Crit,'linewidth',1)
hold
plot(1:nAll,history2.Crit,'linewidth',1)
ylab = ylabel('CV (Logisitc Regression)');
set(ylab,'interpreter','Latex','FontSize',8)
xlab = xlabel('Number of features');
set(xlab,'interpreter','Latex','FontSize',8)
set(gca,'FontSize',6)
leg = legend('Enforcing hierarchical principle', 'Unrestricted');
set(leg,'interpreter','Latex','FontSize',6)
% print('./Figures/eps/WriteUp/featureSelectionLogistic','-depsc')
% print('./Figures/jpegs/WriteUp/featureSelectionLogistic','-djpeg','-r600')

%% Visualizing prediction errors
Xfeatures = X(:,[1;2;3;9;7;8]); %Extracting only features we want

ypred = classify(Xfeatures,Xfeatures,y,'linear'); %This using the entire dataset for training
predLogical = ypred == y; 

videoNum = 1:mAll;
fig2 = figure;%('visible', 'off');
fig2.PaperUnits = 'centimeters';
fig2.PaperPosition = [0 0 8 4];
set(gca,'box','on')
plot(ypred(predLogical),y(predLogical),'k.','markersize',5)
hold
plot(ypred(~predLogical),y(~predLogical),'r.','markersize',5)

ylab = ylabel('Poured volume (c)');
set(ylab,'interpreter','Latex','FontSize',8)
xlab = xlabel('Predicted volume (c)');
set(xlab,'interpreter','Latex','FontSize',8)
set(gca,'FontSize',6,'ylim',[0 3])
%leg = legend('Correct classification', 'Incorrect classification');
%set(leg,'interpreter','Latex','FontSize',6)
% print('./Figures/eps/WriteUp/classificationError','-depsc')
% print('./Figures/jpegs/WriteUp/classificationError','-djpeg','-r600')
