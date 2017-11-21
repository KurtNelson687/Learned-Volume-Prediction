clear;clc;close all;

% files needed to run this:  FitData_usable.mat, QCdata2.mat, S_8516.mat, RedStream_8516.mat
set(0,'DefaultAxesFontSize',18);

% load in feature and label data to be used
load('FitData_usable.mat');
X = Xgood;
y = ygood;
featureDescription1 = {'Duration (s)','Speed (cm/s)','Area (cm^2)',...
                      'Area*Speed*Duration'};
featureDescription2 = {'Duration','Speed','Area'};

% load in data used to created features (pouring start/stop)
load('QCdata2.mat')
load('S_8516.mat')
movN = 80;
Norm = QCdata(movN).SpDiffNorm;
indStart = QCdata(movN).indexes(3);
indEnd   = QCdata(movN).indexes(4)-1;
indMid   = round(mean([indStart,indEnd]));

% load in length scale extraction data
load('RedStream_8516.mat')


%% Plot features against labels
figure;
for i = 1:size(X,2)-1
    subplot(2,2,i);plot(y,X(:,i+1),'.k','markersize',10);hold on;ylabel(featureDescription1{i});
    if i==3
        xlabel('Actual Volume (cups)')
    end
    xlim([0 2.7])
end


%% Plot features against each other
figure;
cnt = 0;
for i = 1:3
    for j = 1:3
        cnt = cnt+1;
        subplot(3,3,cnt);plot(X(:,i+1),X(:,j+1),'.k','markersize',10); % X1 is intercept.
        xlabel([featureDescription2{i}])
        ylabel([featureDescription2{j}])
    end
end

%% Plot norm plot with respective images

figure;
subplot(3,3,1:3);plot(Norm,'k','linewidth',2);hold on;
    plot(indStart,Norm(indStart),'.r','markersize',20) % plot start frame
    plot(indMid,Norm(indMid),'.r','markersize',20)     % plot middle frame
    plot(indEnd,Norm(indEnd),'.r','markersize',20)     % plot end frame
    xlabel('Video Frame #');
    ylabel('|\Delta(Image Intensity)|');
    
subplot(3,3,[4,7]);imshow(S(indStart).cdata);xlabel('Pouring Start')
subplot(3,3,[5,8]);imshow(S(indMid).cdata);xlabel('Mid Pour')
subplot(3,3,[6,9]);imshow(S(indEnd).cdata);xlabel('Pouring End')

%% Plot length scale extraction
figure;
N2 = round(length(RedStreamIm)/2);
    subplot(4,1,1);plot([1,size(Lall,1)],[Ls2,Ls2],'r','linewidth',2);hold on;
                     plot(Lall.^2,'k','linewidth',1);
                     ylabel('Pixel Area')
                     legend('Area_{Movie Ave}','Area_{i^{th} cut, j^{th} Frame}')
                 
    subplot(4,1,2:4);
                   Iplot = RedStreamIm(N2).cdata;
                   imagesc(Iplot);colormap('gray');hold on;
                   for j=1:length(rowi)
                       plot([1,size(Iplot,2)],[rowi(j),rowi(j)],'g')
                       plot([points(N2,j,1),points(N2,j,2)],[rowi(j),rowi(j)],'r','linewidth',3)      
                   end
                   
                  plot([coli,coli],[1,size(Iplot,1)],'g')
                  daspect([1,1,1])
                  xlabel('Pixels')
                  ylabel('Pixels')









