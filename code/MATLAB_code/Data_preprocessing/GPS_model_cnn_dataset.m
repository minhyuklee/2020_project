% GPS model 학습 데이터 생성
clc
clear all
close all
addpath('related_func/')
pathname_train = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\train_data';
pathname_test = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\test_data';
%% Train data load: dataset은 subject별로 데이터 정리, class는 동작별로 데이터 정리 
addpath('raw_dataset/')
dataset = load('raw_gesture_class_train.mat');
dataset = struct2cell(dataset);
dataset = dataset{1,1};
% dataset = 3차원 cell (동작 시행 횟수, 해당 동작 라벨, 피험자번호)

NumOfSubjectForTraining = 16; % 학습에 사용할 피험자 수 (train + validation)

NumOfClass_Gesture = size(dataset{1,2,1},1); % 클래스 수
label_mat = eye(NumOfClass_Gesture);

% 학습에 사용할 데이터 class별로 정리: START
class_training = order_sub2class(dataset,[1,NumOfSubjectForTraining]);
% class_training = 3차원 cell (각 동작의 시행 횟수, 1, 동작번호(클래스))
% 학습에 사용할 데이터 class별로 정리: END
disp('데이터가 클래스 별로 분류되었습니다.')
%% 제스처 데이터 양단 재단
class_training_refine = gesture_length_control(class_training);
%%
g_num = 5;
joint_num = 2;
figure()
for i=1:size(class_training_refine,1)
    plot(class_training_refine{i,1,g_num}(joint_num,:))
    hold on
end
grid on

figure()
for i=1:size(class_training,1)
    plot(class_training{i,1,g_num}(joint_num,:))
    hold on
end
grid on
%% 각 속도 데이터 양단의 offset 보정 구간 확인용
figure()
for i=1:size(velocity_training,1)
    for j=1:size(velocity_training,3)
        plot(velocity_training{i,1,j}(1,1:20),'k');
%         plot(velocity_training{i,1,j}(1,end-19:end),'k');
        hold on
    end
end
% 속도 데이터 양 끝의 12개 값 (총 24개) 값 평균한 만큼 아래로 내리기
%% GPS training dataset generation
class_training_Gesture = GPS_trainset_generation(class_training_refine);
%% 클래스 카테고리 삭제 및 데이터 재정렬
Arrange = class_training_Gesture(:,:,1);
for i=2:size(class_training_Gesture,3)
    Arrange = [Arrange;class_training_Gesture(:,:,i)];
end
%%
figure()
plot(class_training_Gesture{1,2,1})
hold on
plot(class_training_Gesture{1,2,2})
hold on
plot(class_training_Gesture{1,2,3})
hold on
plot(class_training_Gesture{1,2,4})
hold on
plot(class_training_Gesture{1,2,5})
hold on
plot(class_training_Gesture{1,2,6})
hold on
plot(class_training_Gesture{1,2,7})
hold on
plot(class_training_Gesture{1,2,8})
hold on
plot(class_training_Gesture{1,2,9})
hold on
plot(class_training_Gesture{1,2,10})
hold on
plot(class_training_Gesture{1,2,11})
grid on
ylim([-0.2,1.2])
legend('G1','G2','G3','G4','G5','G6','G7','G8','G9','G10','G11')
%% 고정 window size로 한 칸 씩 옮겨가며 데이터 생성
Training_segmentation = {};
w_size = 36;
index = 1;
for i=1:size(Arrange,1)
    gesture_sample = Arrange{i,1};
    GPS_sample = Arrange{i,2};
    sample_length = size(gesture_sample,2);
    for j=1:sample_length-w_size+1
        Training_segmentation{index,1} = gesture_sample(:,j:j+w_size-1);
        Training_segmentation{index,2} = GPS_sample(:,j+w_size-1);
        index = index + 1;
    end
end
%% Scale normalization
[Training_segmentation(:,1,:),stat] = normalize(Training_segmentation(:,1,:));
%%
len_seq = [];
for i=1:size(Training_segmentation,1)
    len_seq = [len_seq,size(Training_segmentation{i,1},2)];
end
max(len_seq)
%% Trainset generation, CSV 파일
check_num = 10^4;

GPStraining_sensor = [];
NumSensor = size(Training_segmentation{i,1},1);
for i=1:size(Training_segmentation,1)
    GPStraining_sensor = [GPStraining_sensor;Training_segmentation{i,1}';check_num*ones(1,NumSensor)];
end
filename = 'GPStraining_sensor.csv';
filedir = fullfile(pathname_train,filename);
csvwrite(filedir,GPStraining_sensor)

GPStraining_GPS = []; % GPStraining_sensor의 각 샘플 내 시퀀스의 마지막 time step에 해당하는 GPS값만 저장
for i=1:size(Training_segmentation,1)
    GPStraining_GPS = [GPStraining_GPS;Training_segmentation{i,2}(:,end);check_num];
end
filename = 'GPStraining_GPS.csv';
filedir = fullfile(pathname_train,filename);
csvwrite(filedir,GPStraining_GPS)
%% Testset generation, mat 파일
% Load raw data
test_sequence = load('raw_gesture_series_test2.mat');
test_sequence = struct2cell(test_sequence);
test_sequence = test_sequence{1,1};

% Normalize & save
test_sequence(1:10,:) = (test_sequence(1:10,:) - stat(:,1)*ones(1,size(test_sequence(1:10,:),2)))./(stat(:,2)*ones(1,size(test_sequence(1:10,:),2)));
filename = 'gesture_series_test2.mat';
filedir = fullfile(pathname_test,filename);
save(filedir,'test_sequence')