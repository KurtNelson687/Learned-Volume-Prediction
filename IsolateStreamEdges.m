% This script isolates the stream by finding the edges of the stream after
% enhancing, finding edges, dilating/smoothing edges, and filling in the
% stream.

% parameters 
WidthThresh = 20; % threshold width of stream

% create structure to store new data
RedStreamIm = struct('cdata',zeros(size(S(100).cdata(:,:,1)),'uint8'),'colormap',[]);
BinaryStreamIm = Sp;
Lsi2 = 0;
TtlPix = 0;

% Get background image to remove
Iback = squeeze(S(QCdata.indexes(2)).cdata(:,:,1));

% % Row indices to pull length scales from
% dr = ceil(size(Iback,1)/3/100);  % row index interval
% rowi  = 5:dr:size(Iback,1)/3;    % ~ 100 rows of data
rowi = ceil(size(Iback,1)/8);  % pull the length scale from 1/8th from top 

% Column limit to search for bright areas
coli = ceil(size(Iback,2)/3);  % chop left third of image

% loop through 1 frames after the start to finish of stream
cnt  = 0; % number of Lengths recorded
L2i  = 0; % initiate L squared
iter = 0; % number of loops performed (index for storing) 
Lall = nan(QCdata.indexes(4)-QCdata.indexes(3),1);
Lrow = nan(QCdata.indexes(4)-QCdata.indexes(3),size(Iback,1));

for i = QCdata.indexes(3)+1:QCdata.indexes(4)
    iter = iter+1;
    
    % pull individual frame
    Imi = squeeze(S(i).cdata(:,:,1)); % Red of RGB image
    
    % Remove background image
    Imi = Iback - Imi;
        
    % Find width of bright areas passed ruler
    ind  = find(Imi(rowi,coli:end)<contrastThresh);
    dind = diff(ind);
    
    % Pull the widest area in the row cross-section (presumably the stream)
    Li   = max(dind);
    
    % Only include widths greater than threshold value
    if Li < WidthThresh
        continue % don't add anything to the mean
    else
        L2i = L2i + Li^2;
        cnt = cnt + 1;
        Lall(iter) = Li;
    end
    
    % Store widths of all bright areas
    Lrow(iter,1:length(dind)) = dind;
    
    % Store Final image
    RedStreamIm(iter).cdata = Imi;

end

Ls2 = L2i/cnt; % the average L^2 (units: pixels^2)

