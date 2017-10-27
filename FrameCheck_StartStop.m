figure;

P = QCdata(1).SpDiffNorm;
N = length(P);

subplot(3,1,1);

for i = 1:N
    clf
    subplot(3,1,1);plot(P);hold on;
    plot(i,P(i),'.r');
    plot(QCdata(1).indexes(3),P(QCdata(1).indexes(3)),'.g','markersize',20)
    plot(QCdata(1).indexes(4),P(QCdata(1).indexes(4)),'.g','markersize',20)
    
    subplot(3,1,2:3);
    imshow(S(i).cdata);
    pause(.01)
end
