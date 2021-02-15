% GPS model 학습 데이터 생성
clc
clear all
close all
addpath('related_func/')
pathname_train = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\train_data';
pathname_test = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\test_data';
%% Train data load: dataset은 subject별로 데이터 정리, class는 동작별로 데이터 정리 
addpath('raw_dataset/')
dataset = load('Access_trainset_G11.mat');
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
%%
mean_length = [];
for i=1:size(class_training_Gesture,3)
    for j=1:size(class_training_Gesture,1)
        mean_length(j,i) = size(class_training_Gesture{j,1,i},2);
    end
end
mean_class = mean(mean_length);
%%
figure()
bar(1:11,mean_class)
set(gca,'xticklabel',{'G1','G2','G3','G4','G5','G6','G7','G8','G9','G10','G11'})
ylabel('Sequence length')
grid on
%% 시퀀스 N분할 v4.0 (각 제스처 3단계 구분, 단계 별로 window size 결정, 각 단계 별 생성하고자하는 데이터 수 입력)
Training_segmentation_sensor = {};
Training_segmentation_GPS = {};
N_early = 10;
N_middle = 10;
N_end = 10;
data_length_early = [];
data_length_middle = [];
data_length_end = [];
phase_length = [];
for i=1:size(Arrange,1)
%     feature_point = gesture_stage_divider(Arrange{i,1});
    feature_point = [];
    feature_point(1,1) = 1;
    for j=1:size(Arrange{i,2},2)
        if Arrange{i,2}(1,j) > 0.1
            feature_point(1,2) = j-1;
            break
        end
    end
    for j=1:size(Arrange{i,2},2)
        if Arrange{i,2}(1,j) > 0.9
            feature_point(1,3) = j-1;
            break
        end
    end
    feature_point(1,4) = size(Arrange{i,2},2);
    
    phase_length(i,1) = feature_point(1,2) - feature_point(1,1) + 1;
    phase_length(i,2) = feature_point(1,3) - feature_point(1,2);
    phase_length(i,3) = feature_point(1,4) - feature_point(1,3);
    
    [segmented_data,interval_location] = moving_window(Arrange{i,1},feature_point,N_early,N_middle,N_end);
    
    % 데이터 길이 분석 단계
%     for j=1:N_early
%         data_length_early = [data_length_early;size(segmented_data{j,1},2)];
%     end
    data_length_early = [data_length_early;phase_length(i,1)];
    
%     for j=N_early+1:N_early+N_middle
%         data_length_middle = [data_length_middle;size(segmented_data{j,1},2)];
%     end
    data_length_middle = [data_length_middle;phase_length(i,2)];
    
%     for j=N_early+N_middle+1:N_early+N_middle+N_end
%         data_length_end = [data_length_end;size(segmented_data{j,1},2)];
%     end
    data_length_end = [data_length_end;phase_length(i,3)];
    
    Training_segmentation_sensor = [Training_segmentation_sensor;segmented_data];
    
    segmented_data_GPS = {};
    for j=1:size(segmented_data,1)
        tmp_location = interval_location{j,1};
        segmented_data_GPS{j,1} = Arrange{i,2}(:,tmp_location);
    end
    Training_segmentation_GPS = [Training_segmentation_GPS;segmented_data_GPS];
end
Training_segmentation = [Training_segmentation_sensor,Training_segmentation_GPS];

% 그래프 그리기
min_early = min(data_length_early);
max_early = max(data_length_early);
min_middle = min(data_length_middle);
max_middle = max(data_length_middle);
min_end = min(data_length_end);
max_end = max(data_length_end);
min_length = min([min_early,min_middle,min_end]);
max_length = max([max_early,max_middle,max_end]);
loop = 1;
for i = min_length:max_length
    length_analyze(loop,:) = [i,sum(data_length_early==i),sum(data_length_middle==i),sum(data_length_end==i)];
    loop = loop + 1;
end

figure()
bar(length_analyze(:,1),length_analyze(:,2:4),'stacked')
set(gca,'FontSize',24)
xlabel('Data length')
ylabel('Count')
xlim([10,70])
% xlim([6,36])
grid on
legend('Preparation','Nucleus','Retraction')
%% 제스처 단계 별 구간 길이 분석 플랏
mean_early = sum(length_analyze(:,1).*length_analyze(:,2))/sum(length_analyze(:,2));
mean_middle = sum(length_analyze(:,1).*length_analyze(:,3))/sum(length_analyze(:,3));
mean_end = sum(length_analyze(:,1).*length_analyze(:,4))/sum(length_analyze(:,4));

mean_2_early = sum(length_analyze(:,1).*length_analyze(:,1).*length_analyze(:,2))/sum(length_analyze(:,2));
mean_2_middle = sum(length_analyze(:,1).*length_analyze(:,1).*length_analyze(:,3))/sum(length_analyze(:,3));
mean_2_end = sum(length_analyze(:,1).*length_analyze(:,1).*length_analyze(:,4))/sum(length_analyze(:,4));

std_early = sqrt(mean_2_early-mean_early^2);
std_middle = sqrt(mean_2_middle-mean_middle^2);
std_end = sqrt(mean_2_end-mean_end^2);

figure()
bar([1,2,3],[mean_early,mean_middle,mean_end],'y')
ylabel('Sequence length')
set(gca,'xticklabel',{'Preparation','Nucleus','Retraction'})
set(gca,'FontSize',16)
hold on
er = errorbar([1,2,3],[mean_early,mean_middle,mean_end],[std_early,std_middle,std_end])
er.Color = [0,0,0];
er.LineStyle = 'none';
hold off

figure()
bar([1,2,3],[mean(phase_length(:,1)),mean(phase_length(:,2)),mean(phase_length(:,3))],'y')
ylabel('Sequence length')
set(gca,'xticklabel',{'Preparation','Nucleus','Retraction'})
set(gca,'FontSize',16)
hold on
er = errorbar([1,2,3],[mean(phase_length(:,1)),mean(phase_length(:,2)),mean(phase_length(:,3))],[std(phase_length(:,1)),std(phase_length(:,2)),std(phase_length(:,3))])
er.Color = [0,0,0];
er.LineStyle = 'none';
hold off
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
NumSensor = size(Training_segmentation{1,1},2);
for i=1:size(Training_segmentation,1)
    GPStraining_sensor = [GPStraining_sensor;Training_segmentation{i,1};check_num*ones(1,NumSensor)];
end
filename = 'Access_GPStraining_sensor.csv';
filedir = fullfile(pathname_train,filename);
csvwrite(filedir,GPStraining_sensor)

GPStraining_GPS = []; % GPStraining_sensor의 각 샘플 내 시퀀스의 마지막 time step에 해당하는 GPS값만 저장
for i=1:size(Training_segmentation,1)
    GPStraining_GPS = [GPStraining_GPS;Training_segmentation{i,2}(:,end);check_num];
end
filename = 'Access_GPStraining_GPS.csv';
filedir = fullfile(pathname_train,filename);
csvwrite(filedir,GPStraining_GPS)
%% Testset generation, mat 파일
% Load raw data
test_sequence = load('raw_gesture_series_test3.mat');
test_sequence = struct2cell(test_sequence);
test_sequence = test_sequence{1,1};

% Normalize & save
test_sequence(1:10,:) = (test_sequence(1:10,:) - stat(:,1)*ones(1,size(test_sequence(1:10,:),2)))./(stat(:,2)*ones(1,size(test_sequence(1:10,:),2)));
filename = 'gesture_series_test3.mat';
filedir = fullfile(pathname_test,filename);
save(filedir,'test_sequence')

% check_num = 10^4;
% GPStest_sensor = [];
% NumSensor = size(test_sequence,1)-1;
% window_size = 35;
% for i=1:size(test_sequence,2)-window_size+1
%     GPStest_sensor = [GPStest_sensor;test_sequence(1:10,i:i+(window_size-1))';check_num*ones(1,NumSensor)];
% end
% csvwrite('GPStest_sensor.csv',GPStest_sensor)
% % serial gesture data를 window size 별로 구분한 데이터