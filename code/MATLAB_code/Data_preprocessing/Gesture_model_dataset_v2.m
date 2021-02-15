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

% NumOfSubjectForTraining = 6; % 학습에 사용할 피험자 수 (train + validation)
% 
% NumOfClass_Gesture = size(dataset{1,2,1},1); % 클래스 수
% label_mat = eye(NumOfClass_Gesture);
% 
% % 학습에 사용할 데이터 class별로 정리: START
% class_training = order_sub2class(dataset,[1,NumOfSubjectForTraining]);
% % class_training = 3차원 cell (각 동작의 시행 횟수, 1, 동작번호(클래스))
% % 학습에 사용할 데이터 class별로 정리: END
% disp('데이터가 클래스 별로 분류되었습니다.')
%% 제스처 데이터 양단 재단
window_size = 20;
sliding = 3;
%% GPS training dataset generation
% 반드시 GPS_model_dataset파일 실행 후에 실행하기
class_training_refine = load('class_training_refine.mat');
class_training_refine = struct2cell(class_training_refine);
class_training_refine = class_training_refine{1,1};
GPS = load('GPS.mat');
GPS = struct2cell(GPS);
GPS = GPS{1,1};
GPS = [GPS,class_training_refine(:,2:3)];
%% GPS 기반으로 Gesture 실행 부분 분리 & Sequence simplification
class_training_Gesture_GPS_thres = cell(size(class_training_refine(:,1)));
class_training_Gesture_GPS_thres(:,2:3) = class_training_refine(:,2:3);
class_training_Gesture_norep = cell(size(class_training_refine(:,1)));
class_training_Gesture_norep(:,2:3) = class_training_refine(:,2:3);
class_training_Gesture = cell(size(class_training_refine));
class_training_Gesture(:,2:3) = class_training_refine(:,2:3);
peak_tol = 25; % degree
tolerance = 0.5;
for i=1:size(GPS,1)
    for j=1:size(GPS,3)
        feature_point = [];
        for k=1:size(GPS{i,1,j},2)
            if GPS{i,1,j}(1,k) > 0.1
                feature_point(1,1) = k-1;
                break
            end
        end
%         feature_point(1,1) = 1;
        for k=1:size(GPS{i,1,j},2)
            if GPS{i,1,j}(1,k) > 0.9
                feature_point(1,2) = k-1;
                break
            end
        end
        class_training_Gesture_GPS_thres{i,1,j} = class_training_refine{i,1,j}(:,feature_point(1):((feature_point(2) - 1)*sliding + window_size));
        
        class_training_Gesture_norep{i,1,j} = repetition_removal(class_training_Gesture_GPS_thres{i,1,j});
        [class_training_Gesture{i,1,j},~] = seq_compress_v2(class_training_Gesture_norep{i,1,j},tolerance);
%         class_training_Gesture{i,1,j} = class_training_Gesture_GPS_thres{i,1,j};
        
        class_training_Gesture{i,1,j} = class_training_Gesture{i,1,j}';
        class_training_Gesture{i,2,j} = class_training_Gesture{i,2,j}';
    end 
end
%%
class_training_Gesture_GPS_thres = cell(size(class_training_refine(:,1)));
class_training_Gesture_GPS_thres(:,2:3) = class_training_refine(:,2:3);
peak_tol = 25; % degree
tolerance = 0.5;
i=81;
close all
figure()
x = 1*(1:1:size(class_training_refine{i,1},2)); % 0.008 default
plot(x,class_training_refine{i,1}','LineWidth',1.5)
grid on
set(gca,'FontSize',16)
xlim([x(1),x(end)])
xlabel('time step')
ylabel('Joint angle (deg)')
for j=1:size(GPS,3)
    feature_point = [];
    for k=1:size(GPS{i,1,j},2)
        if GPS{i,1,j}(1,k) > 0.1
            feature_point(1,1) = k-1;
            break
        end
    end
%         feature_point(1,1) = 1;
    for k=1:size(GPS{i,1,j},2)
        if GPS{i,1,j}(1,k) > 0.9
            feature_point(1,2) = k-1;
            break
        end
    end
    feature_point(1) = 1;
    feature_point(2) = size(GPS{i,1,j},2);
    class_training_Gesture_GPS_thres{i,1,j} = class_training_refine{i,1,j}(:,feature_point(1):((feature_point(2) - 1)*sliding + window_size));

    class_training_Gesture_norep{i,1,j} = repetition_removal(class_training_Gesture_GPS_thres{i,1,j});
%     [class_training_Gesture{i,1,j},~] = seq_compress_v2(class_training_Gesture_norep{i,1,j},tolerance);
%         class_training_Gesture{i,1,j} = class_training_Gesture_GPS_thres{i,1,j};
% 
%     class_training_Gesture{i,1,j} = class_training_Gesture{i,1,j}';
%     class_training_Gesture{i,2,j} = class_training_Gesture{i,2,j}';
end
figure()
x = 1*(1:1:size(class_training_Gesture_norep{i,1,j},2)); % 0.008 default
plot(x,class_training_Gesture_norep{i,1,j}','LineWidth',1.5)
grid on
set(gca,'FontSize',16)
xlim([x(1),x(end)])
xlabel('time step')
ylabel('Joint angle (deg)')
%% 시퀀스 압축 후 확인용 플랏
% 각 제스처 별 길이 줄어든 비율 계산
num_sample = zeros(1,17);
mean_length = zeros(1,17,2);
var_length = zeros(1,17,2);
for i=1:size(class_training_Gesture,1)
    g_ind = find(class_training_Gesture{i,2}==1);
    before_length = size(class_training_Gesture_GPS_thres{i,1},2);
    after_length = size(class_training_Gesture{i,1},1);
    
    % sequence compression 이전의 시퀀스 길이
    mean_length_old = mean_length(1,g_ind,1);
    var_length_old = var_length(1,g_ind,1);
    
    mean_length(1,g_ind,1) = mean_length_old + (before_length - mean_length_old)/(num_sample(g_ind) + 1);
    var_length(1,g_ind,1) = sqrt(var_length_old^2 + mean_length_old^2 - mean_length(1,g_ind,1)*mean_length(1,g_ind,1) + (before_length^2 - var_length_old^2 - mean_length_old^2)/(num_sample(g_ind) + 1));
    
    % sequence compression 이후의 시퀀스 길이
    mean_length_old = mean_length(1,g_ind,2);
    var_length_old = var_length(1,g_ind,2);
    
    mean_length(1,g_ind,2) = mean_length_old + (after_length - mean_length_old)/(num_sample(g_ind) + 1);
    var_length(1,g_ind,2) = sqrt(var_length_old^2 + mean_length_old^2 - mean_length(1,g_ind,2)*mean_length(1,g_ind,2) + (after_length^2 - var_length_old^2 - mean_length_old^2)/(num_sample(g_ind) + 1));    
    
    num_sample(g_ind) = num_sample(g_ind) + 1;
end
std_length = sqrt(var_length);
ratio_length = (mean_length(:,:,1)-mean_length(:,:,2))./mean_length(:,:,1)*100;
mean_ratio_length = mean(ratio_length)

x = categorical({'Pants', 'Milk', 'Who', 'Horse', 'Bird', 'Cry', 'Doubt', 'No', 'Like', 'Want', 'Best','Why','JK','Locate','Look like','Mind freeze','Finish-touch'});
x = reordercats(x,{'Pants', 'Milk', 'Who', 'Horse', 'Bird', 'Cry', 'Doubt', 'No', 'Like', 'Want', 'Best','Why','JK','Locate','Look like','Mind freeze','Finish-touch'});
figure()
h1 = plot(x,mean_length(:,:,1),'--o','Color','r','LineWidth',2);
hold on
er1 = errorbar(x,mean_length(:,:,1),std_length(:,:,1),std_length(:,:,1));
er1.Color = [0 0 0];
er1.LineStyle = 'none';
hold on
h2 = plot(x,mean_length(:,:,2),'--o','Color','b','LineWidth',2);
hold on
er2 = errorbar(x,mean_length(:,:,2),std_length(:,:,2),std_length(:,:,2));
er2.Color = [0 0 0];
er2.LineStyle = 'none';
grid on
xlabel('Gesture index')
ylabel('Average sequence length')
legend([h1, h2],{'Before compression','After compression'})
set(gca,'FontSize',20)
%%
close all
g_num = 17;
finger = 5;
figure(1)
for i=1:size(class_training_Gesture_norep,1)
    if find(class_training_Gesture_norep{i,2}) == g_num
        plot(class_training_Gesture_norep{i,1}(finger,:)')
        hold on
    end
end
set(gca,'FontSize',16)
ylabel('Joint angle')
title('Repetition removal')
grid on

figure(2)
for i=1:size(class_training_Gesture_GPS_thres,1)
    if find(class_training_Gesture_GPS_thres{i,2}) == g_num
        plot(class_training_Gesture_GPS_thres{i,1}(finger,:)')
        hold on
    end
end
set(gca,'FontSize',16)
ylabel('Joint angle')
title('Original repetitive gesture')
grid on

figure(3)
for i=1:size(class_training_Gesture,1)
    if find(class_training_Gesture{i,2}) == g_num
        plot(class_training_Gesture{i,1}(:,finger))
        hold on
    end
end
set(gca,'FontSize',16)
ylabel('Joint angle')
title('Final gesture sequence')
grid on
%% Scale Normalization: train data의 mean/std 구해서 normalize하기
[class_training_Gesture(:,1),stat] = normalize(class_training_Gesture(:,1));
filename = 'RG17_Recognition_stat.mat';
filedir = fullfile(pathname_test,filename);
save(filedir,'stat')
disp('학습 데이터가 정규화되었습니다.')
%% Delete subject index
class_training_Gesture(:,3) = [];
%% Trainset generation, CSV 파일
check_num = 10^4;

% 센서 데이터 csv파일에 맞춰 조정
NumSensor = size(class_training_Gesture{1,1,1},2);
check_num_mat_sensor = repmat({check_num*ones(1,NumSensor)},[size(class_training_Gesture,1),1]);

NumClass = 17;
check_num_mat_class = repmat({check_num*ones(1,NumClass)},[size(class_training_Gesture,1),1]);

check_num_mat = [check_num_mat_sensor, check_num_mat_class];

Gesturetraining = cellfun(@vertcat,class_training_Gesture,check_num_mat,'UniformOutput',false);
Gesturetraining_sensor = cell2mat(Gesturetraining(:,1));
Gesturetraining_class = cell2mat(Gesturetraining(:,2));

% csv파일 생성
filename = 'Gesturetraining_sensor.csv';
filedir = fullfile(pathname_train,filename);
csvwrite(filedir,Gesturetraining_sensor)

filename = 'Gesturetraining_class.csv';
filedir = fullfile(pathname_train,filename);
csvwrite(filedir,Gesturetraining_class)
%% Testset generation, CSV 파일
addpath('raw_dataset/')
% 테스트 데이터 로드
test_data = load('raw_Rgesture_class_test.mat');
test_data = struct2cell(test_data);
test_data = test_data{1,1};

% 오류있는 데이터 제거
% error_ind = [
%             1*ones(2,1),[36;70];...
% %             2*ones(3,1),[27;36;63];...
% %             3*ones(2,1),[36;70];...
%             ]; % col1: subject index , col2: sample index
% subject_ind = cell2mat(test_data(:,3));
% raw_ind = 1:1:size(test_data,1); raw_ind = raw_ind';
% subject_ind = [subject_ind,raw_ind];
% original_ind = [];
% for i=1:size(error_ind,1)
%     tmp = subject_ind(find(subject_ind(:,1)==error_ind(i,1)),:);
%     original_ind = [original_ind, tmp(error_ind(i,2),2)];
% end
% test_data(original_ind,:) = [];
%%
% close all
% clc
% test_input = load('raw_Rgesture_RG2_test.mat');
% test_input = struct2cell(test_input);
% test_input = test_input{1,1};
% % test_input = test_input(:,100:399);
% % x_time = 0.008*(1:1:size(test_input,2));
% figure()
% plot(test_input(1:10,:)')
% grid on
% xlabel('time (sec)')
% ylabel('Joint angle (deg)')
% set(gca,'FontSize',20)
% % test_input_refine{1,1,1} = test_input;
% 
% test_input_refine{1,1,1} = test_input(1:10,1:391);
% test_input_refine{2,1,1} = test_input(1:10,560:924);
% test_input_refine{3,1,1} = test_input(1:10,1160:1576);
% test_input_refine{4,1,1} = test_input(1:10,1783:2236);
% test_input_refine{5,1,1} = test_input(1:10,2417:2975);
% test_input_refine{6,1,1} = test_input(1:10,3201:4053);
%%
window_size = 20;
sliding = 3;
test_input_refine = gesture_length_control_v2(test_data,window_size);
test_input_refine = [test_input_refine,test_data(:,2:3,:)];
[~,GPS_test] = GPS_trainset_generation_v2(test_input_refine(:,1), window_size, sliding);

test_input_Gesture_GPS_thres = cell(size(test_input_refine));
test_input_Gesture_norep = cell(size(test_input_refine));
test_input_Gesture = cell(size(test_input_refine));
peak_tol = 25; % degree
tolerance = 0.5;
for i=1:size(GPS_test,1)
    for j=1:size(GPS_test,3)
        feature_point = [];
        for k=1:size(GPS_test{i,1,j},2)
            if GPS_test{i,1,j}(1,k) > 0.1
                feature_point(1,1) = k-1;
                break
            end
        end
%         feature_point(1,1) = 1;
        for k=1:size(GPS_test{i,1,j},2)
            if GPS_test{i,1,j}(1,k) > 0.9
                feature_point(1,2) = k-1;
                break
            end
        end
        test_input_Gesture_GPS_thres{i,1,j} = test_input_refine{i,1,j}(:,feature_point(1):(feature_point(2)-1)*sliding + window_size);
%         test_input_Gesture_norep{i,1,j} = repetition_removal(test_input_Gesture_GPS_thres{i,1,j});
%         [test_input_Gesture{i,1,j},~] = seq_compress_v2(test_input_Gesture_norep{i,1,j},tolerance);
        test_input_Gesture{i,1,j} = test_input_Gesture_GPS_thres{i,1,j};
        
        test_input_Gesture{i,2,j} = test_data{i,2,j};
    end
end
%%
close all
figure()
plot(test_input_refine{1,1}')
grid on
ylim([-20 100])
ylabel('angle (deg)')
title('Raw data (no rep)')
set(gca,'FontSize',16)
figure()
plot(test_input_Gesture{1,1}')
grid on
ylim([-20 100])
ylabel('angle (deg)')
title('Compressed')
set(gca,'FontSize',16)
figure()
plot(test_input_refine{6,1}')
grid on
ylim([-20 100])
ylabel('angle (deg)')
title('Raw data (5 rep)')
set(gca,'FontSize',16)
figure()
plot(test_input_Gesture{6,1}')
grid on
ylim([-20 100])
ylabel('angle (deg)')
title('Compressed')
set(gca,'FontSize',16)
%%
close all
figure()
subplot(3,1,1)
plot(test_input_Gesture_GPS_thres{1,1,1}')
grid on
% ylabel('joint angle (deg)')
set(gca,'FontSize',24)

subplot(3,1,2)
plot(test_input_Gesture_norep{1,1,1}')
grid on
ylabel('joint angle (deg)')
set(gca,'FontSize',24)

subplot(3,1,3)
plot(test_input_Gesture{1,1,1}')
grid on
% ylabel('joint angle (deg)')
xlabel('data index')
set(gca,'FontSize',24)

figure()
plot(test_input_Gesture{1,1,1}')
grid on
%%
% Save
filename = 'Rgesture_class_test.mat';
filedir = fullfile(pathname_test,filename);
save(filedir,'test_input_Gesture')