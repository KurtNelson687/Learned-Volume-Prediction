function criterion = SVMfit(X_train,y_train,X_test,y_test)
mdl = fitcecoc(X_train,y_train,'verbose',2);
%performs SVM fit using 1 vs all binary learner (each class is postive, and
%all others negative)
ypred = predict(mdl,X_test);           % use model to predict on new test data
criterion = sum(y_test~=ypred);
% specify the criterion to evaluate model performance. error: #wrong/total
end