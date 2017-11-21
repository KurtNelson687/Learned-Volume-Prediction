function criterion = OLSfit(X_train,y_train,X_test,y_test)
mdl = fitlm(X_train,y_train,'linear');
ypred = predict(mdl,X_test);
criterion = sum((y_test-ypred).^2);
end

