clear all
close all
clc
%% Gesture model load (weight와 bias가 csv 파일 형태로 저장된 것을 로드)
% ML_layers 폴더에 저장된 함수 사용
addpath('ML_layers/');
addpath('weights_from_python/Gesture_model');
% Model structure
% Layer1 (index: 0): Input layer
% Layer2 (index: 1): LSTM layer
% Layer3 (index: 2): LSTM layer
% Layer4 (index: 3): Dense layer
% Layer5 (index: 4): Dense layer
% Layer6 (index: 5): Dense layer
% Layer7 (index: 6): Dense layer (output layer, softmax)

% Layer2 (index: 1): LSTM layer
LSTM_weight_x1_raw = csvread('LSTM_weight_x1.csv');
LSTM_weight_h1_raw = csvread('LSTM_weight_h1.csv');
LSTM_bias1_raw = csvread('LSTM_bias1.csv');
units = size(LSTM_weight_x1_raw,2)/4; % 4로 나누는 이유: gate 개수가 총 4개
[LSTM_weight1,LSTM_bias1] = LSTM_mat2cell(LSTM_weight_x1_raw, LSTM_weight_h1_raw, LSTM_bias1_raw,units);

% Layer3 (index: 2): LSTM layer
% LSTM_weight_x2_raw = csvread('LSTM_weight_x2.csv');
% LSTM_weight_h2_raw = csvread('LSTM_weight_h2.csv');
% LSTM_bias2_raw = csvread('LSTM_bias2.csv');
% units = size(LSTM_weight_x2_raw,2)/4; % 4로 나누는 이유: gate 개수가 총 4개
% [LSTM_weight2,LSTM_bias2] = LSTM_mat2cell(LSTM_weight_x2_raw, LSTM_weight_h2_raw, LSTM_bias2_raw,units);

% Layer4 (index: 3): Dense layer
Dense_weight3_raw = csvread('Dense_weight2.csv');
Dense_bias3_raw = csvread('Dense_bias2.csv');
[Dense_weight3, Dense_bias3] = Dense_mat2cell(Dense_weight3_raw, Dense_bias3_raw);

% Layer5 (index: 4): Dense layer
Dense_weight4_raw = csvread('Dense_weight3.csv');
Dense_bias4_raw = csvread('Dense_bias3.csv');
[Dense_weight4, Dense_bias4] = Dense_mat2cell(Dense_weight4_raw, Dense_bias4_raw);

% Layer6 (index: 5): Dense layer
Dense_weight5_raw = csvread('Dense_weight4.csv');
Dense_bias5_raw = csvread('Dense_bias4.csv');
[Dense_weight5, Dense_bias5] = Dense_mat2cell(Dense_weight5_raw, Dense_bias5_raw);

% Layer7 (index: 6): Dense layer (output layer, softmax)
Dense_weight6_raw = csvread('Dense_weight5.csv');
Dense_bias6_raw = csvread('Dense_bias5.csv');
[Dense_weight6, Dense_bias6] = Dense_mat2cell(Dense_weight6_raw, Dense_bias6_raw);

% Save weights as cell array
LSTM_layer_num = [1];
Dense_layer_num = [3,4,5,6];

Directory = "weights_for_MATLAB/Gesture_model";
for i=1:length(LSTM_layer_num)
    save(fullfile(Directory,append("LSTM_weight",num2str(LSTM_layer_num(i)))),append("LSTM_weight",num2str(LSTM_layer_num(i))));
    save(fullfile(Directory,append("LSTM_bias",num2str(LSTM_layer_num(i)))),append("LSTM_bias",num2str(LSTM_layer_num(i))));
end
for i=1:length(Dense_layer_num)
    save(fullfile(Directory,append("Dense_weight",num2str(Dense_layer_num(i)))),append("Dense_weight",num2str(Dense_layer_num(i))));
    save(fullfile(Directory,append("Dense_bias",num2str(Dense_layer_num(i)))),append("Dense_bias",num2str(Dense_layer_num(i))));
end
%% Test dataset 로드 및 모델 구성
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\test_data';
filename = 'Rgesture_class_test.mat';
filedir = fullfile(pathname, filename);
test_dataset = load(filedir);
test_dataset = struct2cell(test_dataset);
test_dataset = test_dataset{1,1};

% Load mean/std for input normalization
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\test_data';
filename = 'RG17_Recognition_stat.mat';
filedir = fullfile(pathname,filename);
Recognition_stat = load(filedir);
Recognition_stat = struct2cell(Recognition_stat);
Recognition_stat = Recognition_stat{1,1};

peak_tol = 25;
tolerance = 0.5;
processing_time = [];
for i=1:size(test_dataset,1)
    input = test_dataset{i,1};
    tic
    
    input = repetition_removal(input);
    input = seq_compress_v2(input,tolerance);
    input_norm = (input - Recognition_stat(1,:)'*ones(1,size(input,2)))./(Recognition_stat(2,:)'*ones(1,size(input,2)));
    input_norm = input_norm';

    [LSTM_hidden1,~] = LSTM_forward(input_norm,LSTM_weight1,LSTM_bias1);

    [Dense_hidden1,~] = Dense_forward(LSTM_hidden1(end,:), Dense_weight3, Dense_bias3, 'ReLU');
    [Dense_hidden2,~] = Dense_forward(Dense_hidden1, Dense_weight4, Dense_bias4, 'ReLU');
    [Dense_hidden3,~] = Dense_forward(Dense_hidden2, Dense_weight5, Dense_bias5, 'ReLU');

    [output,~] = Dense_forward(Dense_hidden3, Dense_weight6, Dense_bias6, 'softmax');
    
    processing_time = [processing_time,toc];
    test_output{i,1} = output;

    true_output{i,1} = test_dataset{i,2}';
end
%% Processing time
p_time_gesture = zeros(size(processing_time,2),17);
for i=1:size(processing_time,2)
    g_num = find(true_output{i,1} == 1);
    p_time_gesture(i,g_num) = processing_time(i);
end

p_time_grouping = {};
mean_processing = [];
for i=1:17
    p_time_grouping{1,i} = p_time_gesture(find(p_time_gesture(:,i)),i);
    mean_processing(1,i) = mean(p_time_grouping{1,i});
end
figure()
bar(mean_processing)
%% Plot confusion matrix
% matrix expansion
true_label_mat = [];
test_label_mat = [];
for i=1:size(true_output,1)
    true_label_mat = [true_label_mat,true_output{i,1}'];
    test_label_mat = [test_label_mat,test_output{i,1}'];
end

% confusion matrix
cm = plotconfusion(true_label_mat,test_label_mat);
set(gca,'xticklabel',{'Pants', 'Milk', 'Who', 'Horse', 'Bird', 'Cry', 'Doubt', 'No', 'Like', 'Want', 'Best','Why','JK','Locate','Look like','Mind freeze','Finish-touch','Acc'})
set(gca,'yticklabel',{'Pants', 'Milk', 'Who', 'Horse', 'Bird', 'Cry', 'Doubt', 'No', 'Like', 'Want', 'Best','Why','JK','Locate','Look like','Mind freeze','Finish-touch','Acc'})
%%
% 반복 제스처 3번 반복 (G1~G8), 비반복 제스처 (G9~G17)
no_compression_pc_time = 10^3*[0.0458, 0.0371, 0.0325, 0.0334, 0.0301, 0.0300, 0.0316, 0.0307, 0.0054, 0.0092, 0.0098, 0.0093, 0.0104, 0.0070, 0.0102, 0.0084, 0.0080];
compression_pc_time = 10^3*[0.0214, 0.0074, 0.0046, 0.0059, 0.0062, 0.0037, 0.0041, 0.0040, 0.0026, 0.0033, 0.0036, 0.0037, 0.0043, 0.0037, 0.0045, 0.0037, 0.0046];
reduced_ratio = (no_compression_pc_time - compression_pc_time)./no_compression_pc_time * 100;
reduced_ratio_repetitive = mean(reduced_ratio(1:8))
reduced_ratio_non_repetitive = mean(reduced_ratio(9:17))
figure()
x = categorical({'Pants', 'Milk', 'Who', 'Horse', 'Bird', 'Cry', 'Doubt', 'No', 'Like', 'Want', 'Best','Why','JK','Locate','Look like','Mind freeze','Finish-touch'});
x = reordercats(x,{'Pants', 'Milk', 'Who', 'Horse', 'Bird', 'Cry', 'Doubt', 'No', 'Like', 'Want', 'Best','Why','JK','Locate','Look like','Mind freeze','Finish-touch'});
bar(x,[no_compression_pc_time',compression_pc_time'])
grid on
ylabel('processing (ms)')
legend('w/o compression','w/ compression')
set(gca,'FontSize',26)