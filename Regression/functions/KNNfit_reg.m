function criterion = KNNfit_reg(X_train,y_train,X_test,y_test)

% find the nearest neighbor in X_train for each point in X_test
Ind = knnsearch(X_train,X_test,'k',1);

ypred = y_train(Ind);

ypred = mean(ypred,1);

criterion = sum((y_test(:)-ypred(:)).^2);    % specify the criterion to evaluate model performance. sum of squared error.
end
