% this script finds the indices for the frames containing the stream flow
% start and stop.


%% Finds indicies for when stream enters and leaves view
indBefore = find(QCdata(movieNum).SpDiffNorm>valPaperThresh,1,'first'); %random index when paper is infront of camera
indFirstLow = indBefore+find(QCdata(movieNum).SpDiffNorm(indBefore:end)<valMinThresh,1,'first')-1;

NormRunMin = zeros(length(QCdata(movieNum).SpDiffNorm),1);
for i = indFirstLow:length(QCdata(movieNum).SpDiffNorm)-numStartThresh
    NormRunMin(i) = min(QCdata(movieNum).SpDiffNorm(i:i+numStartThresh-1));
end

indS1 = find(NormRunMin>streamStartThresh,1,'first');
indS2 = indS1+1;

indE1 = indS1+find(QCdata(movieNum).SpDiffNorm(indS1:end)<streamEndThresh,1,'first')-1;

QCdata(movieNum).indexes = [indBefore, indFirstLow, indS1, indE1];

%% Saves images around first and last stream appearance for QC
QCdata(movieNum).Im0 = S(indS1-1).cdata;
QCdata(movieNum).Im1 = S(indS1).cdata;
QCdata(movieNum).Im2 = S(indS2).cdata;

QCdata(movieNum).ImE0 = S(indE1-1).cdata;
QCdata(movieNum).ImE1 = S(indE1).cdata;
QCdata(movieNum).ImE2 = S(indE1+1).cdata;
