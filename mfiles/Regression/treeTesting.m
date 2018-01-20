% Script: treeTesting.m
%
% Author: Kurt Nelson and Sam Maticka
%
% Purpose: This script computes the training RMSE using a boosted
% regression tree. This has not been fully tested. Beware if using this
% script. 
%%
close all; clear all;
load('../../../DataFiles/data.mat')
addpath('../../Functions/')

%% Variables
numTrees = 150; % number of trees 
maxNumSplits = 100; % maximum number of splits per tree
learningRate = 0.1; % learning rate for boosted tree
Xfeatures = 1:10; % features to keep in tree

%% Setup data
X=X_train(:,2:end);
y=y_train';

% Adds columns to X so that all second order terms of original features are included
X = addInteractions(X);
X = X(:,Xfeatures);

%% Fit boosted tree
[m,n] = size(X);
t = templateTree('MaxNumSplits',maxNumSplits);
Mdl = fitensemble(X,y,'LSBoost',numTrees,t,...
    'Type','regression','KFold',10,'LearnRate',learningRate);
MSE = kfoldLoss(Mdl);
