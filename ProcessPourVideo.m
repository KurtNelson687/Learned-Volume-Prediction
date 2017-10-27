clear;clc;close all;
tic
% Adjustments needed: 
% 1. still camera to remove baseline background (find baseline after find initial stream)
% 2. constant color background to remove easy (matte black, no glare),
% 3. add a penny to the background for pixel to real size calibration 

% algorithm essentials:
% 1: find first frame of pour (correlation between consecutive images will
% be constant until water begins to pour).
% 2: take the median of all frames before the pouring frame and remove from
% all frames
% 3: Use the first pouring frame and following one to calculate falling speed:
% we have dt from frame rate, and can get dx by finding vertical distance between 
% bottom edge in 2 images. maybe cross-correlating the images through vertical displacements. Once the edges 'line up' that
% number of pixels is converted to distance. Assume same falling rate for
% all pouring time. 
% 4: for each frame 

%Flags
plotImages = false; %plot images
saveQCdata = true; %save QC data
saveFeatures = true; % save X and y, which contain features and responses

Nave = 5; %number of frames to averge over for background subtraction

%Threshold values for finding start and stop of stream
streamStartThresh = 6; %threshold for defining stream start
numStartThresh = 10; %number of forward frames streamStartThresh must be exceeded
valPaperThresh = 25; %threshold for finding a frame when the paper label is moving
valMinThresh = 2; %threshold for finding a point after the paper label leaves view but before stream
streamEndThresh = 1.8; %threshold for when stream ends

%Threshold value for increasing contast
contrastThresh = 12;


video_folder = './FirstFilmSesssion_Oct24/'; %folder storing videos
movies = dir([video_folder '*.MOV']); %all *.MOV files  

%for movieNum = 1:numel(movies)
for movieNum = 1
disp(['Moive ' num2str(movieNum) ' out of ' num2str(numel(movies))])
vidObj = VideoReader([video_folder movies(movieNum).name]);
fps = vidObj.FrameRate; % frames/second
dt  = 1/fps;            % seconds between frames
N   = floor(vidObj.Duration/dt); %number of frames


% create movie structure array to store stills
S = struct('cdata',zeros(vidObj.Height,vidObj.Width,3,'uint8'),'colormap',[]);
Sp = S;
SpDiffNorm = zeros(1,numel(Sp));
k = 1;

while hasFrame(vidObj)
    S(k).cdata = readFrame(vidObj);
    Sp(k).cdata = imcomplement(rgb2gray(S(k).cdata));
    
    if k>1
        QCdata(movieNum).SpDiffNorm(k) =  norm(im2double(Sp(k).cdata)-...
            im2double(Sp(k-1).cdata)); %2-norm of differnce between two grayscale frames
    end
    
    k = k+1;
end

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

% Make more contrasted. 
Im1en = Im1o;
Im2en = Im2o;
Im1en(Im1o>=contrastThresh) = 150;Im1en(Im1o<contrastThresh) = 0;
Im2en(Im2o>=contrastThresh) = 150;Im1en(Im2o<contrastThresh) = 0;

%% Find Edges
BW1 = edge(Im1en,'Canny',.95);
BW2 = edge(Im2en,'Canny',.95);

%% Dilate the image
se90 = strel('line', 3, 90);
se0  = strel('line', 3, 0);

BW1dil = imdilate(BW1, [se90 se0]);
BW2dil = imdilate(BW2, [se90 se0]);

BW1dil(1,:)=true;
BW2dil(1,:)=true;

%% fill holes
BW1fill = imfill(BW1dil, 'holes');
BW2fill = imfill(BW2dil, 'holes');
%% Save images for QC check of front speed calculation
QCdata(movieNum).BW1fill = BW1fill;
QCdata(movieNum).BW2fill = BW2fill;
%% Compute front speed
%__________Need to find length per pixel______
lenPerPix = 1;
%_____________________________________________
[m1,n1] = find(BW1fill==1);
[m2,n2] = find(BW2fill==1);
pixDif = max(m2)-max(m1);
frontSpeed = pixDif*lenPerPix/dt;

%% Create feature matrix X and output vector y
X(movieNum,1) = (indE1-indS1)*dt; %first column of X is stream duration
X(movieNum,2) = frontSpeed; %second column of X is front speed
X(movieNum,3) = NaN; %third column of X is a representative length scale
X(movieNum,4) = NaN; %fourth column of X total occupied pixels during pouring after threshold

%%__________Fill y vector______
y(movieNum) = NaN;
%%______________________________


if plotImages
figure;
subplot(1,2,1);imshow(QCdata(movieNum).Im1);
title('original images')
subplot(1,2,2);imshow(QCdata(movieNum).Im2);

figure;imshow(ImAve);title('5-frame Average')

figure;
subplot(1,6,1);imshow(QCdata(movieNum).Im2);title('Original Image Inverted')
subplot(1,6,2);imshow(Im2o);title('Background Removed')
subplot(1,6,3);imshow(Im2en);title('Enhanced')
subplot(1,6,4);imshow(BW2);title('Canny Edge Detection of Enhanced')
subplot(1,6,5);imshow(BW2dil);title('Dilated Canny Edges')
subplot(1,6,6);imshow(BW2fill);title('Filled Edges')
end
%%
end
toc

if saveQCdata
save('QCdata.mat','QCdata')
end

if saveFeatures
save('FitData.mat','QCdata')
end



