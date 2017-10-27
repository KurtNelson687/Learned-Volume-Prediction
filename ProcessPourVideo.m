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

% create video object with structure of data about movie
vidObj = VideoReader('./FirstFilmSesssion_Oct24/IMG_8363.MOV');

fps = vidObj.FrameRate; % frames/second
dt  = 1/fps;            % seconds between frames
N   = vidObj.Duration/dt;

% Get individual frames of movie

% create movie structure array to store stills
S = struct('cdata',zeros(vidObj.Height,vidObj.Width,3,'uint8'),'colormap',[]);
k = 1; 
while hasFrame(vidObj)
    S(k).cdata = readFrame(vidObj);
    k = k+1;
end

% Manually Find first and second image of water stream to process speed
% choose single image to process 
ind1 = 67;
ind2 = ind1+1;

Im1 = imcomplement(rgb2gray(S(ind1).cdata));
Im2 = imcomplement(rgb2gray(S(ind2).cdata));

figure;
subplot(1,2,1);imshow(Im1);
title('original images')
subplot(1,2,2);imshow(Im2);

% Find average of N shots before the start
Nave = 5;
s = nan(Nave,size(Im1,1),size(Im1,2));
for i = 1:Nave
    s(i,:,:) = imcomplement(rgb2gray(S(ind1-i).cdata));
end
ImAve = uint8(squeeze(mean(s,1)));

figure;imshow(ImAve);title('5-frame Average')

% remove average from each still frame
Sp = S;
for i = 1:size(Sp,2)
    Sp(i).cdata = imcomplement(rgb2gray(S(i).cdata)) - ImAve;
end

% background removed
Im1o = squeeze(Sp(ind1).cdata);
Im2o = squeeze(Sp(ind2).cdata);

% Make more contrasted. 
Im1en = Im1o;
Im2en = Im2o;
Thresh = 5;
Im1en(Im1o>=Thresh) = 150;Im1en(Im1o<Thresh) = 0;
Im2en(Im2o>=Thresh) = 150;Im1en(Im2o<Thresh) = 0;

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

figure;
subplot(1,6,1);imshow(Im2);title('Original Image Inverted')
subplot(1,6,2);imshow(Im2o);title('Background Removed')
subplot(1,6,3);imshow(Im2en);title('Enhanced')
subplot(1,6,4);imshow(BW2);title('Canny Edge Detection of Enhanced')
subplot(1,6,5);imshow(BW2dil);title('Dilated Canny Edges')
subplot(1,6,6);imshow(BW2fill);title('Filled Edges')

%%
toc




