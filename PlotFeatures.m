% this script plots each feature against video number (which should be
% increasing with volume)
close all; clear;
load('FitData_KN.mat')
load('movieLabel.mat')

saveGoodData = true; % choose whether to save reduced data set

%badData gives movie numbers that will be removed. 
badData = [8443; 8446; 8447; 8457; 8458;...
    8461; 8462; 8463; 8467; 8472; 8476;...
    8486; 8489; 8490; 8496; 8502;...
    8507; 8509; 8510; 8511; 8513; 8520;...
    8524; 8530; 8538; 8559; 8571];

%Errors associated with a incorrect ruler, front velocity, and length scale are below 
%Bad ruler = 8458; 8461; 8462; 8472; 8496; 8507; 8509; 8510 (bad start too)
%Bad velocity = 8476; 8486; 8489; 8490; 8520 (bad start too); 8559;
%Bad length scale = 8513; 8511; 8463; 8446;
%% Data Cleaning - finds indicies for good and bad data
count =1;
count2 = 1;
for i = 1:length(movieLabel)
    if isempty(find(badData==movieLabel(i),1))
        goodInd(count)=i;
        count=count+1;
    else
        badInd(count2)=i;
        count2 = count2+1;
    end
end
% create subsets of data. useable and nonusable 
Xbad = X(badInd,:);
ybad = y(badInd);

Xgood = X(goodInd,:);
ygood = y(goodInd);
MovLabelgood = movieLabel(goodInd);

%Replace bad data with NaN's - keep NaN values in though to retrain
%proper indexing for identify bad features. 
X(badInd,:) = NaN;
y(badInd)   = NaN;

cuppcm3 = .00422675;      % cups per cm^3
Vol     = pi/4*X(:,4)*cuppcm3;

scale = nanmean(y'./Vol); % Scaling for physics prediction
yPhs  = scale*Vol;

featureDescription = {'Duration (s)','Speed (cm/s)','L^2_{ave} (cm^2)',...
                      'L^2*Speed*Duration'};

%% Feature plots against label
fig1 = figure;
for i = 1:size(X,2)-1
    subplot(5,1,i);plot(X(:,i),'k');ylabel(featureDescription{i});
end
subplot(5,1,4);plot((Vol'-y)./y,'k');ylabel('% Error for Vol')
subplot(5,1,5);plot((yPhs'-y)./y,'k');ylabel('% Error for Scaled Vol')
xlabel('Movie Number')

fig2 = figure;
for i = 1:size(X,2)
    subplot(2,2,i);plot(y,X(:,i),'*k');hold on;ylabel(featureDescription{i});
                   plot(ybad,Xbad(:,i),'*r');
end
legend('Good Data','Bad Data','location','NW')
xlabel('Actual Volume (cups)')

%% Data fittig

%Removing NaN's for Ordinary least squares prediction
X = X(goodInd,:);
y = y(goodInd);
yPhs = yPhs(goodInd);

X = [ones(size(X,1),1),X];
cuppcm3 = .00422675; % cups per cm^3
X(:,end) = pi/4*X(:,end)*cuppcm3;

% Ordinary least squares optimum theta
theta = inv(X'*X)*X'*y';

yhat = X*theta;

%Percent error calculations
OLSerror = abs(yhat - y')./y'*100;
meanOLSerror = mean(OLSerror);

physicsError = abs(yPhs - y')./y'*100; 
meanPhysicsError = mean(physicsError);


%% Prediction plots
fig3 = figure;
plot(y,yhat,'.r',y,yPhs,'.b',[0, 2.5],[0 2.5],'k','markersize',10)
xlabel('Measured Volume (cups)'); ylabel('Predicted Volume (cups)')
leg = legend('Physics', 'OLS prediction');

%% plot Features against each other

figure;
cnt = 0;
for i = 1:4
    for j = 1:4
        cnt = cnt+1;
        subplot(4,4,cnt);plot(X(:,i+1),X(:,j+1),'.k'); % X1 is intercept.
        xlabel(['X',num2str(i)])
        ylabel(['X',num2str(j)])
    end
end


%% save good data set
if saveGoodData
    % add intercept term
    Xgood = [ones(size(Xgood(:,1))),Xgood];
    save('FitData_usable.mat','Xgood','ygood','MovLabelgood')
end



