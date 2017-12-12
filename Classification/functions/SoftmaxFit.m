function criterion = SoftmaxFit(X_train,y_train,X_test,y_test)
% for input data:
% X is nxp. n observations, p features
% y is nx1. n observations. each value is scalar from 1:k. k possible categories.

%Splits and recombines data so training set has two of each label
[X_train,y_train,X_test,y_test] = combineThenSplit(X_train,y_train,X_test,y_test);

% fit coefficients, B for a multinomial logistic regression model
B = mnrfit(X_train,y_train,'model','ordinal'); 
% assumes natural ordering among the response (ytrain) categories.
% B is (p+1)x(k?1). includes intercept.

% calculates probability for each observation to be 1 of k categories.
prob = mnrval(B,X_test,'model','ordinal'); 
% prob is nxk

% Choose category with highest probability
[~,ypred] = max(prob,[],2); % column index corresponds to category index

% error: ratio of wrongly categorized data
criterion = sum(y_test(:)~=ypred(:));
end
