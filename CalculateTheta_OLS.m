% Calculate theta
clear;clc;close all;

load('FitData.mat')

X = [ones(size(X,1),1),X];
cuppcm3 = .00422675; % cups per cm^3

X(:,end) = pi/4*X(:,end)*cuppcm3;
% Ordinary least squares optimum theta
theta = inv(X'*X)*X'*y';

yhat = X*theta;

figure;plot(y,yhat,'.-k')

error = sum(((yhat - y')./y).^2);
