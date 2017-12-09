% weighted least squares fit
function criterion = wt_percentDiff_fit(X_train,y_train,X_test,y_test)
% train and create a linear regression model using the expected value as the weight
mdl = fitlm(X_train,y_train,'linear','weights',y_train); 
ypred = predict(mdl,X_test);         % use model to predict on new test data
criterion = sum((y_test-ypred).^2);  % specify the criterion to evaluate model performance. sum of squared error.
end

