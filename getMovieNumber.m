% This code is modelled after './ProcessPourVideo.m'
% it takes a list of movies and creates a feature set and response vector

% Video Folder location
%video_folder = './FirstFilmSesssion_Oct24/'; %folder storing videos

%% Locate Videos to be processed
%movies = dir([video_folder '*.MOV']); % all *.MOV files

for movieNum = 1:numel(movies)

movieLabel(movieNum)=str2num(movies(movieNum).name(end-7:end-4));

end