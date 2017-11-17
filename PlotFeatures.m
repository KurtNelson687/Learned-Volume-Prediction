% this script plots each feature against video number (which should be
% increasing with volume)
close all; clear all;
load('FitData.mat')
load('moveLabel.mat')

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

%Replace bad data with NaN's
X(badInd,:)=NaN;
y(badInd)=NaN;

cuppcm3 = .00422675; % cups per cm^3
Vol = pi/4*X(:,4)*cuppcm3;

%% Feature plots
fig1 = figure;
subplot(5,1,1);plot(X(:,1),'k');ylabel('Duration (s)');
subplot(5,1,2);plot(X(:,2),'k');ylabel('Speed (cm/s)')
subplot(5,1,3);plot(X(:,3),'k');ylabel('L^2_{ave} (cm^2)')
subplot(5,1,4);plot((Vol'-y)./y,'k');ylabel('% Error for Vol')

scale = nanmean(y'./Vol); %Scaling for physics prediction
yPhs = scale*Vol;

subplot(5,1,5);plot((yPhs'-y)./y,'k');ylabel('% Error for Scaled Vol')
xlabel('Movie Number')

fig2 = figure;
subplot(3,1,1);plot(y,X(:,1),'*k');ylabel('Duration (s)')
subplot(3,1,2);plot(y,X(:,2),'*k');ylabel('Speed (cm/s)')
subplot(3,1,3);plot(y,X(:,3),'*k');ylabel('L^2_{ave} (cm^2)');xlabel('cups');


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
