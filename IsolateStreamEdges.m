% This script isolates the stream by finding the edges of the stream after
% enhancing, finding edges, dilating/smoothing edges, and filling in the
% stream.

% create structure to store new data
BinaryStreamIm = Sp;
Lsi2 = 0;
TtlPix = 0;

for i = 1:size(Sp,2)
    % pull individual frame
    Imi = squeeze(Sp(i).cdata);
    
    % enhance by making black or white
    Imi(Imi>=contrastThresh) = 256; % black  
    Imi(Imi<contrastThresh)  = 0;   % white

    % Find Edges
    Imi = edge(Imi,'Canny',.95);
    
    % dilate edges
    Imi = imdilate(Imi, [se90 se0]);

    % make whole first row true so a hole is detected
    Imi(1,:)=true;
    
    % fill in the holes
    Imi = imfill(Imi, 'holes');
    
    % count number of white (1) pixels in each row
    Lsi = sum(Imi(2:end,:),2); % don't include 1st row. false 1s.

    % sum and square the length
    Lsi2 = Lsi2 + mean(Lsi.^2);
    
    % count total number of pixels
    TtlPix = TtlPix + sum(sum(Imi(2:end,:)));

    % Store Final image
    BinaryStreamIm(i).cdata = Imi;

end

Ls2 = Lsi2/i; % the average of averages of L^2 (units: pixels^2)

% Make more contrasted. 
Im1en = Im1o;
Im2en = Im2o;
Im1en(Im1o>=contrastThresh) = 256;    Im1en(Im1o<contrastThresh) = 0;
Im2en(Im2o>=contrastThresh) = 256;    Im2en(Im2o<contrastThresh) = 0;
% 0:   black
% 256: white (stream and noise)


%% Find Edges
BW1 = edge(Im1en,'Canny',.95);
BW2 = edge(Im2en,'Canny',.95);

%% Dilate the image (for 2 images, this section takes .7 seconds)
BW1dil = imdilate(BW1, [se90 se0]);
BW2dil = imdilate(BW2, [se90 se0]);

BW1dil(1,:)=true;
BW2dil(1,:)=true;

%% fill holes
BW1fill = imfill(BW1dil, 'holes');
BW2fill = imfill(BW2dil, 'holes');

%% Calculate width of stream per row (in pixel units)
% count number of white (1) pixels in each row
Ls21 = sum(BW1fill(2:end,:),2); % don't include 1st row. false 1s.
Ls22 = sum(BW2fill(2:end,:),2);

% Square the value and take average per frame
Ls21save = mean(Ls21.^2);
Ls22save = mean(Ls22.^2);


