clear all
close all
clc
%% GPS model_cnn_v2 load (weight와 bias가 csv 파일 형태로 저장된 것을 로드)
% ML_layers 폴더에 저장된 함수 사용
addpath('ML_layers/');
addpath('weights_from_python/GPS_model');
% Model structure
% Layer1 (index: 0): Input layer
% Layer2 (index: 1): Conv1D layer
% Layer3 (index: 2): Maxpooling1D layer
% Layer4 (index: 3): Flatten layer
% Layer5 (index: 4): Dense layer
% Layer6 (index: 5): Dense layer
% Layer7 (index: 6): Dense layer
% Layer8 (index: 7): Dense layer
% Layer9 (index: 8): Dense layer
% Layer10 (index: 9): Dense layer
% Layer11 (index: 10): Dense layer (output layer)

% Layer2 (index: 1): Conv1D layer
Conv1D_weight1_raw = csvread('Conv1D_weight1.csv');
Conv1D_bias1_raw = csvread('Conv1D_bias1.csv');
Conv1D_kernel_size = 21;
[Conv1D_weight1, Conv1D_bias1] = Conv1D_mat2cell(Conv1D_weight1_raw, Conv1D_bias1_raw, Conv1D_kernel_size);

% Layer5 (index: 4): Dense layer
Dense_weight4_raw = csvread('Dense_weight4.csv');
Dense_bias4_raw = csvread('Dense_bias4.csv');
[Dense_weight4, Dense_bias4] = Dense_mat2cell(Dense_weight4_raw, Dense_bias4_raw);

% Layer6 (index: 5): Dense layer
Dense_weight5_raw = csvread('Dense_weight6.csv');
Dense_bias5_raw = csvread('Dense_bias6.csv');
[Dense_weight5, Dense_bias5] = Dense_mat2cell(Dense_weight5_raw, Dense_bias5_raw);

% Layer7 (index: 6): Dense layer (output layer)
Dense_weight6_raw = csvread('Dense_weight8.csv');
Dense_bias6_raw = csvread('Dense_bias8.csv');
[Dense_weight6, Dense_bias6] = Dense_mat2cell(Dense_weight6_raw, Dense_bias6_raw);

% Layer8 (index: 7): Dense layer
Dense_weight7_raw = csvread('Dense_weight10.csv');
Dense_bias7_raw = csvread('Dense_bias10.csv');
[Dense_weight7, Dense_bias7] = Dense_mat2cell(Dense_weight7_raw, Dense_bias7_raw);

% Layer9 (index: 8): Dense layer
Dense_weight8_raw = csvread('Dense_weight12.csv');
Dense_bias8_raw = csvread('Dense_bias12.csv');
[Dense_weight8, Dense_bias8] = Dense_mat2cell(Dense_weight8_raw, Dense_bias8_raw);

% Layer10 (index: 9): Dense layer
Dense_weight9_raw = csvread('Dense_weight14.csv');
Dense_bias9_raw = csvread('Dense_bias14.csv');
[Dense_weight9, Dense_bias9] = Dense_mat2cell(Dense_weight9_raw, Dense_bias9_raw);

% Layer11 (index: 10): Dense layer (output layer)
Dense_weight10_raw = csvread('Dense_weight16.csv');
Dense_bias10_raw = csvread('Dense_bias16.csv');
[Dense_weight10, Dense_bias10] = Dense_mat2cell(Dense_weight10_raw, Dense_bias10_raw);

% Save weights as cell array
Conv1D_layer_num = [1];
Dense_layer_num = [4,5,6,7,8,9,10];

Directory = "weights_for_MATLAB";
for i=1:length(Conv1D_layer_num)
    save(fullfile(Directory,append("Conv1D_weight",num2str(Conv1D_layer_num(i)))),append("Conv1D_weight",num2str(Conv1D_layer_num(i))));
    save(fullfile(Directory,append("Conv1D_bias",num2str(Conv1D_layer_num(i)))),append("Conv1D_bias",num2str(Conv1D_layer_num(i))));
end
for i=1:length(Dense_layer_num)
    save(fullfile(Directory,append("Dense_weight",num2str(Dense_layer_num(i)))),append("Dense_weight",num2str(Dense_layer_num(i))));
    save(fullfile(Directory,append("Dense_bias",num2str(Dense_layer_num(i)))),append("Dense_bias",num2str(Dense_layer_num(i))));
end
% %%
% train_input = csvread('GPStraining_sensor.csv');
% train_input = train_input(1:30,:);
% train_input = [zeros(6,10);train_input];
%% Test dataset 로드 및 모델 구성
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\test_data';
filename = 'gesture_series_test3.mat';
filedir = fullfile(pathname,filename);
test_dataset = load(filedir);
test_dataset = struct2cell(test_dataset);
test_dataset = test_dataset{1,1};

test_data_sensor = test_dataset(1:10,:);
test_data_true_label = test_dataset(11,:);

window_size = 36;
test_output = [];

for t=1:size(test_data_sensor,2)-window_size+1
    input = test_data_sensor(:,t:t+window_size-1)';
%     input = train_input;
    
    Conv1D_hidden1 = Conv1D_forward(input, Conv1D_weight1, Conv1D_bias1, 1, 'ReLU');
    Conv1D_hidden1 = MaxPooling1D(Conv1D_hidden1, 2);
    Conv1D_hidden1 = Flatten(Conv1D_hidden1);

    [Dense_hidden1,~] = Dense_forward(Conv1D_hidden1, Dense_weight4, Dense_bias4, 'ReLU');
    [Dense_hidden2,~] = Dense_forward(Dense_hidden1, Dense_weight5, Dense_bias5, 'ReLU');
    [Dense_hidden3,~] = Dense_forward(Dense_hidden2, Dense_weight6, Dense_bias6, 'ReLU');
    [Dense_hidden4,~] = Dense_forward(Dense_hidden3, Dense_weight7, Dense_bias7, 'ReLU');
    [Dense_hidden5,~] = Dense_forward(Dense_hidden4, Dense_weight8, Dense_bias8, 'ReLU');
    [Dense_hidden6,~] = Dense_forward(Dense_hidden5, Dense_weight9, Dense_bias9, 'ReLU');
    
    [test_output(:,t),~] = Dense_forward(Dense_hidden6, Dense_weight10, Dense_bias10, 'sigmoid');
end
%%
figure()
plot(test_output, 'b', 'LineWidth', 2)
hold on
plot(test_data_true_label(1,end-size(test_output,2)+1:end), 'g', 'LineWidth', 2)
hold on
plot(0.9*ones(1,length(test_output)), 'r--', 'LineWidth', 2)
hold on
plot(0.1*ones(1,length(test_output)), 'r--', 'LineWidth', 2)
grid on
ylim([-0.2 1.2])
legend('Prediction', 'Ground truth', 'GPS=0.9', 'GPS=0.1')
set(gca,'FontSize',15)
%%
figure()
plot(test_data_sensor')