clear;clc;close all;
load('../DataFiles/data.mat')
addpath('./functions');

figure;
for ii=1:2
    X = X_train(:,2:end);
    
    if ii==1
        % normalize
        X = X./repmat(max(X),size(X,1),1);
    end
    
    [coef,scores,pcvars] = princomp(X);
    
    % The representation of X in principal component space
    c1=scores(:,1); % 1st component
    c2=scores(:,2); % 2nd component
    c3=scores(:,3); % 3rd component
    c4=scores(:,4); % 4th component
    
    % The variance explained by each of the 4 components (pcvars are the
    % eigenvalues of the covariance matrix of X)
    var1 = pcvars(1)/sum(pcvars)*100;
    var2 = pcvars(2)/sum(pcvars)*100;
    var3 = pcvars(3)/sum(pcvars)*100;
    var4 = pcvars(4)/sum(pcvars)*100;
    
    subplot(1,2,ii);scatter(c1,c2,25,y_train,'filled')
    axis square
    xlabel(['PC1 (var = ',num2str(var1),')'])
    ylabel(['PC2 (var = ',num2str(var2),')'])
    
    if ii==1
        title('X normalized')
    else
        title('X not normalized')
    end
    co=colorbar;
    ylabel(co,'Actual volume (c)')
    
end