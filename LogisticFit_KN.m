function criterion = LogisticFit_KN(X_train,y_train,X_test,y_test)
 ypred = classify(X_test,X_train,y_train,'linear'); %this can be used for
% what I believe is softmax

%lda = fitcdiscr(X_train,y_train,'DiscrimType','linear'); %Use this for different discriminant models
%[ypred,~,~] = predict(lda,X_test);

criterion = sum(~(y_test==ypred));
end

