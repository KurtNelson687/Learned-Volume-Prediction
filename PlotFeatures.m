% this script plots each feature against video number (which should be
% increasing with volume)

load('FitData.mat')


%%
cuppcm3 = .00422675; % cups per cm^3
Vol = pi/4*X(:,4)*cuppcm3;

figure;
subplot(4,1,1);plot(X(:,1),'k');ylabel('Duration (s)')
subplot(4,1,2);plot(X(:,2),'k');ylabel('Speed (cm/s)')
subplot(4,1,3);plot(X(:,3),'k');ylabel('L^2_{ave} (cm^2)')
subplot(4,1,4);plot(Vol,'k');ylabel('Volume Proxy (cups)')
               xlabel('Movie Number')



