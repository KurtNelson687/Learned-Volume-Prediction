clear;clc;close all;
load('FitData_usable.mat')

% create possible feature sets
% Rename and remove intercept feature
X=Xgood(:,2:end);
y=ygood;
y = y';
y = y*4; % make an integer from 1 to k

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
[fs1,history1] = sequentialfs(@LogisticFit_SM,X,y,'cv',numKfold,...
    'keepin',featuresIn,'nfeatures',nAll,'options',opts);
sse1 = min(history1.Crit); % sum of squared error from the best fit (sse=mse, since 1 test point)
% Sequential feature selection: performs cross-val on numKfolds, forces
% alg. to keep in the 3 fundamental features, makes sure the alg. performs
% over all possible features (doesn't stop at a local min), and display
% information at each sequential iteration. 

%Perform feature selection with no requirments
[fs2,history2] = sequentialfs(@LogisticFit_SM,X,y,'cv',numKfold,...
    'nfeatures',nAll,'options',opts);
sse2 = min(history2.Crit); % sum of squared error from the best fit

% sse1 = 0.2692. sse2 = 0.2692.
%% choose features based on selection above
Xfeatures = X(:,[1;2;3;6;10]); %Extracting only features we want
% second lowest sse yielding feature set chosen to not overfit

%% use chosen features to create logistic regression model
B = mnrfit(Xfeatures,y,'model','ordinal'); % multinomial logistic regression
% assumes natural ordering among the response (ytrain) categories.
% B is (p+1)x(k?1). includes intercept.

% calculates probability for each observation to be 1 of k categories.
prob = mnrval(B,Xfeatures,'model','ordinal'); 
% prob is nxk

% Choose category with highest probability
[~,ypred] = max(prob,[],2); % column index corresponds to category index

predLogical = y~=ypred;
error = sum(predLogical)/length(ypred); % error: # wrong/total #
       
%% plot stuff

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


videoNum = 1:mAll;
fig2 = figure;%('visible', 'off');
fig2.PaperUnits = 'centimeters';
fig2.PaperPosition = [0 0 8 4];
set(gca,'box','on')
plot(ypred/4,y/4,'k.','markersize',5)
hold on;
plot(ypred(predLogical)/4,y(predLogical)/4,'r.','markersize',5)

title(['Multinomial Logistic Regression. Training error: ',num2str(error*100),'%'])
ylab = ylabel('Poured volume (c)');
set(ylab,'interpreter','Latex','FontSize',8)
xlab = xlabel('Predicted volume (c)');
set(xlab,'interpreter','Latex','FontSize',8)
set(gca,'FontSize',6,'ylim',[0 3])
%leg = legend('Correct classification', 'Incorrect classification');
%set(leg,'interpreter','Latex','FontSize',6)
% print('./Figures/eps/WriteUp/classificationError','-depsc')
% print('./Figures/jpegs/WriteUp/classificationError','-djpeg','-r600')

fig3 = figure;%('visible', 'off');
fig3.PaperUnits = 'centimeters';
fig3.PaperPosition = [0 0 8 4];
set(gca,'box','on')

plot(y(~predLogical),X(~predLogical,4),'k.','markersize',5)
hold
plot(y(predLogical),X(predLogical,4),'r.','markersize',5)

ylab = ylabel('physics volume');
set(ylab,'interpreter','Latex','FontSize',8)
xlab = xlabel('actual volume');
set(xlab,'interpreter','Latex','FontSize',8)
set(gca,'FontSize',6)
leg = legend('correct classification', 'misclassification');
set(leg,'interpreter','Latex','FontSize',6)
print('./Figures/eps/WriteUp/classificationError2','-depsc')
print('./Figures/jpegs/WriteUp/classificationError2','-djpeg','-r600')


