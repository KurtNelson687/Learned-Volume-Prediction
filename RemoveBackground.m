% this script finds the average of Nave frames prior to the initial
% stream-containing frame and removes it from all frames.

%% Find average of N shots before the start
s = nan(Nave,size(QCdata(movieNum).Im1,1),size(QCdata(movieNum).Im1,2));
for i = 1:Nave
    s(i,:,:) = imcomplement(rgb2gray(S(indS1-i).cdata));
end
ImAve = uint8(squeeze(mean(s,1)));

% remove average from each still frame
for i = 1:size(Sp,2)
    Sp(i).cdata = Sp(i).cdata - ImAve;
end

% background removed
Im1o = squeeze(Sp(indS1).cdata);
Im2o = squeeze(Sp(indS2).cdata);
