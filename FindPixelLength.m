running = 30;
redTol = 50;

red  = S(QCdata(movieNum).indexes(2)).cdata(:,:,1);


[m,n] = size(red);

clear xRightNans temp
for i = 1:m
    for j =1+running:n
        runningMaxBack(i,j) = max(red(i,j-running:j));
    end
end

xRight = zeros(m,1);
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

tapeColumnInd = xRight(tape2Start);