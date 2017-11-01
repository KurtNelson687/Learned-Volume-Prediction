running = 30;
redTol = 50;

red  = S(QCdata(movieNum).indexes(2)).cdata(:,:,1);


[m,n] = size(red);

for i = 1:m
    for j =1+running:n
        runningMaxBack(i,j) = max(red(i,j-running:j));
    end
end

for i = 1:m
    if(isempty(find(runningMaxBack(i,running+1:end)<redTol,1,'last')))
        xRight(i) = NaN;
    else
        xRight(i) = find(runningMaxBack(i,running+1:end)<redTol,1,'last')+running;
    end
end

%temp = xRight;
% for i = find(~isnan(xRight))
%     if abs(temp(i)-temp(i-1))>1 || abs(temp(i)-temp(i+1))>1
%     xRight(i) = NaN;
%     end
% end

temp(isnan(xRight)) = -1;
tape1Start = find(temp == 0,1,'first');
xRight(xRight<nanmean(xRight(tape1Start+100:tape1Start+150))-25)=NaN;

xRightNans(isnan(xRight)) = -1;
tape1End = find(xRightNans(tape1Start+100:end) == -1,1,'first')+tape1Start+98;
tape2Start = find(xRightNans(tape1End+100:end) == 0,1,'first')+tape1End+99;

numPix = sqrt((xRight(tape1End)-xRight(tape2Start))^2+(tape1End-tape2Start)^2);
lenPerPix = rulerLength/numPix;

fig = figure;
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 8 7];
set(gca,'box','on')
clf
ha = tight_subplot(1,2,[0.03 .02],[.07 .04],[.06 .05]);

axes(ha(1))
imshow(S(QCdata(movieNum).indexes(2)).cdata)
hold
plot([xRight(tape1End) xRight(tape2Start)],[tape1End tape2Start],'b','linewidth',1)

axes(ha(2))
imshow(runningMaxBack)
hold
plot(xRight,1:length(xRight),'r','linewidth',1)

clear xRightNans xRight temp

% print(['./Figures/eps/RulerCheck/RulerLength' num2str(movieNum)],'-depsc')
% print(['./Figures/jpegs/RulerCheck/RulerLength' num2str(movieNum)],'-djpeg','-r600')
