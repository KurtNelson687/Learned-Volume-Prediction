% This code is modelled after './ProcessPourVideo.m'
% it takes a list of movies and creates a feature set and response vector
clear;clc;close all

tic
% 1 Add description
% 2 Convert to function? flages, movie location, make feature set, and
%   plotting outside of function. 
% 3 in each sub-script, turn into function and only output needed variables 
% 4 to make faster, don't store all edge frames, just store feature info from each

%% Define parameters for processing video
% Flags
plotImages   = false; % plot images
saveQCdata   = true;  % save QC data
saveFeatures = true;  % save X and y, which contain features and responses

Nave = 5; % number of frames to averge over for background subtraction

% Threshold values for finding start and stop of stream
streamStartThresh = 6;   % threshold: defining stream start
numStartThresh    = 10;  % # of forward frames streamStartThresh must be exceeded
valPaperThresh    = 25;  % threshold: finding frame when paper label is moving
valMinThresh      = 2;   % threshold: finding point after paper label leaves view but before stream
streamEndThresh   = 1.8; % threshold: when stream ends

% Values for ruler calculation
rulerLength = 4;
redTol = 50;
running = 30;

% Threshold value for increasing contrast
contrastThresh = 15;
% threshold width of stream
WidthThresh = 10;  %%%%%% change this to cm and add upper limit in isolateStream


% parameters for dilation
se90 = strel('line', 3, 90);
se0  = strel('line', 3, 0);


% Video Folder location
video_folder = '../../FirstFilmSesssion_Oct24/'; %folder storing videos


%% Locate Videos to be processed
movies = dir([video_folder '*.MOV']); % all *.MOV files  


%%

for movieNum = 1:numel(movies)
% for movieNum = 1  %%%%%%%uncomment this and comment next line when done
%movieNum = 1;

PullFramesFromMov; 
% stores frames in a structure (S: grayscale; Sp: invert of S)
% QCdata.SpDiffNorm: norm of difference between consecutive frames


FindStartStopIndices;
% Finds 1st 2 frames of stream in view and last frame of stream in view
% indS1, indS2, indE1, respectively.
% QCdata(movieNum).indexes: contains indices
% Saves frames buffering stream start and stop events.
% QCdata(movieNum).Im## holds each image

RemoveBackground;
% Rewrites Sp after removing background from each frame
% stores 1st 2 stream frames Im1o and Im2o for plotting

IsolateStreamEdges;
% average length scale (pixels) squared: Ls2 
% Final images are stored in new structure, RedStreamIm


%% Save images for QC check of front speed calculation
QCdata(movieNum).BW1fill = BW1fill;
QCdata(movieNum).BW2fill = BW2fill;

%% Finds length of pixel
FindPixelLength
% Finds length of pixel

%% Compute front speed
[m1,n1] = find(BW1fill==1);
[m2,n2] = find(BW2fill==1);
pixDif  = max(m2)-max(m1);
frontSpeed = pixDif*lenPerPix/dt;


%% Convert length squared to true length
Ls2true = Ls2*lenPerPix^2;

%% Calculate Volume Proxy (interaction term)
Vol = Ls2true*frontSpeed*(indE1-indS1)*dt;


%% Create feature matrix X and output vector y
X(movieNum,1) = (indE1-indS1)*dt; %first column of X is stream duration
X(movieNum,2) = frontSpeed; %second column of X is front speed
X(movieNum,3) = Ls2true; % third column of X is a representative length scale
X(movieNum,4) = Vol; %fourth column of X theoretical volume estimate 

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
     save('FitData.mat','X','y')
end
