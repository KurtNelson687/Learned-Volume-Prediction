function criterion = Ridgefit(X_train,y_train,X_test,y_test)
tuning = 0.31;
beta = ridge(y_train,X_train,tuning,0);
ypred = X_test*beta(2:end)+beta(1);
criterion = sum((y_test-ypred).^2);    % specify the criterion to evaluate model performance. sum of squared error.
end

