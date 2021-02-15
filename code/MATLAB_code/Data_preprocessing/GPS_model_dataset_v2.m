% GPS model 학습 데이터 생성
clc
clear all
close all
addpath('related_func/')
pathname_train = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\train_data';
pathname_test = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\test_data';
%% Train data load: dataset은 subject별로 데이터 정리, class는 동작별로 데이터 정리 
addpath('raw_dataset/')
dataset = load('raw_Rgesture_class_train.mat');
dataset = struct2cell(dataset);
dataset = dataset{1,1};
% dataset = 3차원 cell (동작 시행 횟수, 해당 동작 라벨, 피험자번호)

% NumOfSubjectForTraining = 1; % 학습에 사용할 피험자 수 (train + validation)
% 
% NumOfClass_Gesture = size(dataset{1,2,1},1); % 클래스 수
% label_mat = eye(NumOfClass_Gesture);
% 
% % 학습에 사용할 데이터 class별로 정리: START
% class_training = order_sub2class(dataset,[1,NumOfSubjectForTraining]);
% % class_training = 3차원 cell (각 동작의 시행 횟수, 1, 동작번호(클래스))
% % 학습에 사용할 데이터 class별로 정리: END
% disp('데이터가 클래스 별로 분류되었습니다.')
%% 오류있는 데이터 제거
error_ind = [
            2*ones(1,1),[81];...
            6*ones(1,1),[56];...
            7*ones(7,1),[11;43;46;68;69;70;75];...
            8*ones(4,1),[1;10;22;84];...
            9*ones(6,1),[10;19;21;32;56;78];...
            10*ones(6,1),[11;15;27;46;54;79];...
            11*ones(3,1),[16;57;61];...
            12*ones(3,1),[27;36;63];...
            13*ones(2,1),[36;70];...
            ]; % col1: subject index , col2: sample index
subject_ind = cell2mat(dataset(:,3));
raw_ind = 1:1:size(dataset,1); raw_ind = raw_ind';
subject_ind = [subject_ind,raw_ind];
original_ind = [];
for i=1:size(error_ind,1)
    tmp = subject_ind(find(subject_ind(:,1)==error_ind(i,1)),:);
    original_ind = [original_ind, tmp(error_ind(i,2),2)];
end
dataset(original_ind,:) = [];
%% 제스처 데이터 양단 재단
window_size = 20;
class_training_refine = gesture_length_control_v2(dataset,window_size);
class_training_refine = [class_training_refine,dataset(:,2:3,:)];

% save('class_training_refine.mat','class_training_refine')
%% GPS training dataset generation
sliding = 3;
[class_training_sample,class_training_GPS] = GPS_trainset_generation_v2(class_training_refine(:,1), window_size, sliding);
% save('GPS.mat','class_training_GPS')
class_training_GPS = [class_training_GPS,class_training_refine(:,2:3)];
%%
close all
g_num = 14;
sub_num = 14;

subject_ind = cell2mat(class_training_refine(:,3));
selected_subject = find(subject_ind==sub_num);
plotting_angle = class_training_refine(selected_subject,:,:);
plotting_GPS = class_training_GPS(selected_subject,:,:);

figure(11)
for i=1:size(plotting_GPS,1)
    if find(plotting_GPS{i,2}) == g_num
        plot(plotting_GPS{i,1})
        hold on
        grid on
    end
end

% for finger=1:10
%     figure(finger)
%     for i=1:size(plotting_angle,1)
%         if find(plotting_angle{i,2}) == g_num
%             plot(plotting_angle{i,1}(finger,:))
%             hold on
%             grid on
%         end
%     end
% end
%% Scale normalization
[class_training_sample(:,1),stat] = normalize(class_training_sample(:,1));
filename = 'RG17_Spotting_stat.mat';
filedir = fullfile(pathname_test,filename);
save(filedir,'stat')
%% Trainset generation, CSV 파일
check_num = 10^4;

NumSensor = size(class_training_sample{1,1},2);
check_num_mat_sensor = repmat({check_num*ones(1,NumSensor)},[size(class_training_sample,1),1]);

NumOut = 1;
check_num_mat_out = repmat({check_num*ones(1,NumOut)},[size(class_training_sample,1),1]);

check_num_mat = [check_num_mat_sensor,check_num_mat_out];

GPStraining = cellfun(@vertcat,class_training_sample,check_num_mat,'UniformOutput',false);
GPStraining_sensor = cell2mat(GPStraining(:,1));
GPStraining_GPS = cell2mat(GPStraining(:,2));

filename = 'GPStraining_sensor.csv';
filedir = fullfile(pathname_train,filename);
csvwrite(filedir,GPStraining_sensor)

filename = 'GPStraining_GPS.csv';
filedir = fullfile(pathname_train,filename);
csvwrite(filedir,GPStraining_GPS)
%% Testset generation, mat 파일
addpath('raw_dataset/')
% Load raw data
% test_sequence = load('for_plotting.mat');
test_sequence = load('raw_Rgesture_series_test.mat');
test_sequence = struct2cell(test_sequence);
test_sequence = test_sequence{1,1};

% Save the data
% filename = 'for_plotting.mat';
filename = 'Rgesture_series_test.mat';
filedir = fullfile(pathname_test,filename);
save(filedir,'test_sequence')