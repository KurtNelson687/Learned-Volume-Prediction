function criterion = LDAfit(X_train,y_train,X_test,y_test)
%  ypred = classify(X_test,X_train,y_train,'linear'); % GDA, assumes normal distribution

%Splits and recombines data so training set has two of each label
[X_train,y_train,X_test,y_test] = combineThenSplit(X_train,y_train,X_test,y_test);

lda = fitcdiscr(X_train,y_train,'DiscrimType','linear'); %  GDA, assumes normal distribution
[ypred,~,~] = predict(lda,X_test);

criterion = sum(y_test(:)~=ypred(:));
end

