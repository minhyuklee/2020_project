clc
clear all
close all
addpath('related_func/');
pathname_train = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\train_data';
pathname_test = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\test_data';
%% Train data load: dataset�� subject���� ������ ����, class�� ���ۺ��� ������ ����
addpath('raw_dataset/')
dataset = load('Access_trainset_G11.mat');
dataset = struct2cell(dataset);
dataset = dataset{1,1};
% dataset = 3���� cell (���� ���� Ƚ��, �ش� ���� ��, �����ڹ�ȣ)

NumOfSubjectForTraining = 16; % �н��� ����� ������ �� (train + validation)

NumOfClass_Gesture = size(dataset{1,2,1},1); % Ŭ���� ��
label_mat = eye(NumOfClass_Gesture);

% �н��� ����� ������ class���� ����: START
class_training = order_sub2class(dataset,[1,NumOfSubjectForTraining]);
% class_training = 3���� cell (�� ������ ���� Ƚ��, 1, ���۹�ȣ(Ŭ����))
% �н��� ����� ������ class���� ����: END
disp('�����Ͱ� Ŭ���� ���� �з��Ǿ����ϴ�.')
%% ����ó ������ ��� ���
class_training_refine = gesture_length_control(class_training);
%%
g_num = 1;
joint_num = 4;
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
%% �� �ӵ� ������ ����� offset ���� ���� Ȯ�ο�
figure()
for i=1:size(velocity_training,1)
    for j=1:size(velocity_training,3)
        plot(velocity_training{i,1,j}(1,1:20),'k');
%         plot(velocity_training{i,1,j}(1,end-19:end),'k');
        hold on
    end
end
% �ӵ� ������ �� ���� 12�� �� (�� 24��) �� ����� ��ŭ �Ʒ��� ������
%% GPS training dataset generation
class_training_Gesture = GPS_trainset_generation(class_training_refine);
%% GPS ������� Gesture ���� �κ� �и� & Sequence simplification
tolerance = 0.5;
for i=1:size(class_training_Gesture,1)
    for j=1:size(class_training_Gesture,3)
        feature_point = [];
        feature_point(1,1) = 1;
        for k=1:size(class_training_Gesture{i,2,j},2)
            if class_training_Gesture{i,2,j}(1,k) > 0.9
                feature_point(1,2) = k;
                break
            end
        end
        class_training_Gesture{i,3,j} = class_training_Gesture{i,1,j}(:,feature_point(1):feature_point(2));
        class_training_Gesture{i,3,j} = seq_compress_v2(class_training_Gesture{i,3,j},tolerance);
        class_training_Gesture{i,3,j} = class_training_Gesture{i,3,j}';
    end
end
class_training_Gesture(:,1,:) = class_training_Gesture(:,3,:);
class_training_Gesture(:,3,:) = [];
class_training_Gesture(:,2,:) = [];
%% ������ ���� �� Ȯ�ο� �ö�
figure()
plot(class_training_Gesture{25,1,2}(3,:))
grid on
%% Scale Normalization: train data�� mean/std ���ؼ� normalize�ϱ�
[class_training_Gesture,stat] = normalize(class_training_Gesture);
% stat�� ù column�� mean, �ι�° column�� std
disp('�н� �����Ͱ� ����ȭ�Ǿ����ϴ�.')
%% Trainset generation, CSV ����
check_num = 10^4;

Gesturetraining_sensor = [];
Gesturetraining_class = [];
NumSensor = size(class_training_Gesture{1,1,1},2);
NumClass = 11;
class_mat = eye(NumClass);
for i=1:size(class_training_Gesture,3)
    for j=1:size(class_training_Gesture,1)
        Gesturetraining_sensor = [Gesturetraining_sensor;class_training_Gesture{j,:,i};check_num*ones(1,NumSensor)];
        Gesturetraining_class = [Gesturetraining_class;class_mat(i,:);check_num*ones(1,NumClass)];
    end
end
filename = 'Access_Gesturetraining_sensor.csv';
filedir = fullfile(pathname_train,filename);
csvwrite(filedir,Gesturetraining_sensor)

filename = 'Access_Gesturetraining_class.csv';
filedir = fullfile(pathname_train,filename);
csvwrite(filedir,Gesturetraining_class)
%% Testset generation, CSV ����
% �׽�Ʈ ������ �ε�
test_data = load('Access_testset_G11.mat');
test_data = struct2cell(test_data);
test_data = test_data{1,1};
NumOfSubjectForTest = 4; % �н��� ����� ������ �� (train + validation)

NumOfClass_Gesture = size(test_data{1,2,1},1); % Ŭ���� ��
label_mat = eye(NumOfClass_Gesture);
class_test = order_sub2class(test_data,[1,NumOfSubjectForTest]);

% Sequence simplification
test_input = class_test;
test_input_refine = gesture_length_control(test_input);
test_input_Gesture = GPS_trainset_generation(test_input_refine);
tolerance = 0.5;
for i=1:size(test_input_Gesture,1)
    for j=1:size(test_input_Gesture,3)
        feature_point = [];
        feature_point(1,1) = 1;
        for k=1:size(test_input_Gesture{i,2,j},2)
            if test_input_Gesture{i,2,j}(1,k) > 0.9
                feature_point(1,2) = k;
                break
            end
        end
        test_input_Gesture{i,3,j} = test_input_Gesture{i,1,j}(:,feature_point(1):feature_point(2));
        test_input_Gesture{i,3,j} = seq_compress_v2(test_input_Gesture{i,3,j},tolerance);
        test_input_Gesture{i,3,j} = test_input_Gesture{i,3,j}';
    end
end
test_input_Gesture(:,1,:) = test_input_Gesture(:,3,:);
test_input_Gesture(:,3,:) = [];
test_input_Gesture(:,2,:) = [];

% Normalization
for i=1:size(test_input_Gesture,3)
    for j=1:size(test_input_Gesture,1)
        test_input_Gesture{j,:,i} = (test_input_Gesture{j,:,i} - stat(1,:)*ones(size(test_input_Gesture,2),1))./(stat(2,:)*ones(size(test_input_Gesture,2),1));
    end
end
filename = 'Access_gesture_class_test.mat';
filedir = fullfile(pathname_test,filename);
save(filedir,'test_input_Gesture')

% 
% check_num = 10^4;
% Gesturetest_sensor = [];
% Gesturetest_class = [];
% NumSensor = size(test_input_Gesture{1,1,1},1);
% NumClass = 11;
% class_mat = eye(NumClass);
% for i=1:size(test_input_Gesture,3)
%     for j=1:size(test_input_Gesture,1)
%         Gesturetest_sensor = [Gesturetest_sensor;test_input_Gesture{j,:,i}';check_num*ones(1,NumSensor)];
%         Gesturetest_class = [Gesturetest_class;class_mat(i,:);check_num*ones(1,NumClass)];
%     end
% end
% csvwrite('Gesturetest_sensor.csv',Gesturetest_sensor)
% csvwrite('Gesturetest_class.csv',Gesturetest_class)