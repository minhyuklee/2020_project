clear all
close all
clc
%% GPS model_v2 load (weight와 bias가 csv 파일 형태로 저장된 것을 로드)
addpath('ML_layers/');
addpath('weights_from_python/GPS_model');
% Model structure
% Layer1 (index: 0): Input layer
% Layer2 (index: 1): Masking layer
% Layer3 (index: 2): Bi-LSTM layer
% Layer4 (index: 3): Dense layer
% Layer5 (index: 4): Dense layer
% Layer6 (index: 5): Dense layer
% Layer7 (index: 6): Dense layer
% Layer8 (index: 7): Dense layer
% Layer9 (index: 8): Dense layer
% Layer10 (index: 9): Dense layer (output layer)

% Layer3 (index: 2): Bi-LSTM layer
BiLSTM_weight_x2_forward_raw = csvread('LSTM_weight_x2_forward.csv');
BiLSTM_weight_h2_forward_raw = csvread('LSTM_weight_h2_forward.csv');
BiLSTM_bias2_forward_raw = csvread('LSTM_bias2_forward.csv');
BiLSTM_weight_x2_backward_raw = csvread('LSTM_weight_x2_backward.csv');
BiLSTM_weight_h2_backward_raw = csvread('LSTM_weight_h2_backward.csv');
BiLSTM_bias2_backward_raw = csvread('LSTM_bias2_backward.csv');
units = size(BiLSTM_weight_x2_forward_raw,2)/4; % 4로 나누는 이유: gate 개수가 총 4개
[BiLSTM_weight2,BiLSTM_bias2] = BiLSTM_mat2cell(BiLSTM_weight_x2_forward_raw, BiLSTM_weight_h2_forward_raw, BiLSTM_bias2_forward_raw, BiLSTM_weight_x2_backward_raw, BiLSTM_weight_h2_backward_raw, BiLSTM_bias2_backward_raw, units);

% Layer4 (index: 3): Dense layer
Dense_weight3_raw = csvread('Dense_weight3.csv');
Dense_bias3_raw = csvread('Dense_bias3.csv');
[Dense_weight3, Dense_bias3] = Dense_mat2cell(Dense_weight3_raw, Dense_bias3_raw);

% Layer5 (index: 4): Dense layer
Dense_weight4_raw = csvread('Dense_weight5.csv');
Dense_bias4_raw = csvread('Dense_bias5.csv');
[Dense_weight4, Dense_bias4] = Dense_mat2cell(Dense_weight4_raw, Dense_bias4_raw);

% Layer6 (index: 5): Dense layer
Dense_weight5_raw = csvread('Dense_weight7.csv');
Dense_bias5_raw = csvread('Dense_bias7.csv');
[Dense_weight5, Dense_bias5] = Dense_mat2cell(Dense_weight5_raw, Dense_bias5_raw);

% Layer7 (index: 6): Dense layer
Dense_weight6_raw = csvread('Dense_weight9.csv');
Dense_bias6_raw = csvread('Dense_bias9.csv');
[Dense_weight6, Dense_bias6] = Dense_mat2cell(Dense_weight6_raw, Dense_bias6_raw);

% Layer8 (index: 7): Dense layer
Dense_weight7_raw = csvread('Dense_weight11.csv');
Dense_bias7_raw = csvread('Dense_bias11.csv');
[Dense_weight7, Dense_bias7] = Dense_mat2cell(Dense_weight7_raw, Dense_bias7_raw);

% Layer9 (index: 8): Dense layer
Dense_weight8_raw = csvread('Dense_weight13.csv');
Dense_bias8_raw = csvread('Dense_bias13.csv');
[Dense_weight8, Dense_bias8] = Dense_mat2cell(Dense_weight8_raw, Dense_bias8_raw);

% Layer10 (index: 9): Dense layer (output layer)
Dense_weight9_raw = csvread('Dense_weight15.csv');
Dense_bias9_raw = csvread('Dense_bias15.csv');
[Dense_weight9, Dense_bias9] = Dense_mat2cell(Dense_weight9_raw, Dense_bias9_raw);

% Save weights as cell array
BiLSTM_layer_num = [2];
Dense_layer_num = [3,4,5,6,7,8,9];

Directory = "weights_for_MATLAB/GPS_model";
for i=1:length(BiLSTM_layer_num)
    save(fullfile(Directory,append("BiLSTM_weight",num2str(BiLSTM_layer_num(i)))),append("BiLSTM_weight",num2str(BiLSTM_layer_num(i))));
    save(fullfile(Directory,append("BiLSTM_bias",num2str(BiLSTM_layer_num(i)))),append("BiLSTM_bias",num2str(BiLSTM_layer_num(i))));
end
for i=1:length(Dense_layer_num)
    save(fullfile(Directory,append("Dense_weight",num2str(Dense_layer_num(i)))),append("Dense_weight",num2str(Dense_layer_num(i))));
    save(fullfile(Directory,append("Dense_bias",num2str(Dense_layer_num(i)))),append("Dense_bias",num2str(Dense_layer_num(i))));
end
%% Test dataset 로드 및 모델 구성
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\test_data';
% filename = 'for_plotting.mat';
filename = 'Rgesture_series_test.mat';
filedir = fullfile(pathname,filename);
test_dataset = load(filedir);
test_dataset = struct2cell(test_dataset);
test_dataset = test_dataset{1,1};

% Load mean/std for input normalization
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\test_data';
filename = 'RG17_Spotting_stat.mat';
filedir = fullfile(pathname,filename);
Spotting_stat = load(filedir);
Spotting_stat = struct2cell(Spotting_stat);
Spotting_stat = Spotting_stat{1,1};

test_data_sensor = test_dataset(1:10,:);
test_data_true_label = test_dataset(11,:);

window_size = 20;
test_output = [];
save_concat = [];
for t=1:size(test_data_sensor,2)-window_size+1
    input = test_data_sensor(:,t:t+window_size-1);
    input_norm = (input - Spotting_stat(1,:)'*ones(1,size(input,2)))./(Spotting_stat(2,:)'*ones(1,size(input,2)));
    input_norm = input_norm';
    
    BiLSTM_hidden1 = BiLSTM_forward(input_norm,BiLSTM_weight2,BiLSTM_bias2,'False','concat');

    [Dense_hidden1,~] = Dense_forward(BiLSTM_hidden1(end,:), Dense_weight3, Dense_bias3, 'ReLU');
    [Dense_hidden2,~] = Dense_forward(Dense_hidden1, Dense_weight4, Dense_bias4, 'ReLU');
    [Dense_hidden3,~] = Dense_forward(Dense_hidden2, Dense_weight5, Dense_bias5, 'ReLU');
    [Dense_hidden4,~] = Dense_forward(Dense_hidden3, Dense_weight6, Dense_bias6, 'ReLU');
    [Dense_hidden5,~] = Dense_forward(Dense_hidden4, Dense_weight7, Dense_bias7, 'ReLU');
    [Dense_hidden6,~] = Dense_forward(Dense_hidden5, Dense_weight8, Dense_bias8, 'ReLU');
    
    [test_output(:,t),~] = Dense_forward(Dense_hidden6, Dense_weight9, Dense_bias9, 'sigmoid');
end
%%
figure()
subplot(2,1,1)
x_time = 0.008*(0:500);
plot(x_time,test_data_sensor(:,1:501)')
grid on
ylabel('joint angle (deg)')
set(gca,'FontSize',24)

subplot(2,1,2)
x_time_GPS = 0.008*(39:500);
plot(x_time_GPS,test_output(:,1:462), 'b', 'LineWidth', 2)
hold on
plot(x_time,0.9*ones(1,length(x_time)), 'r--', 'LineWidth', 2)
hold on
plot(x_time,0.1*ones(1,length(x_time)), 'r--', 'LineWidth', 2)
grid on
ylim([-0.2 1.2])
xlabel('time (sec)')
ylabel('GPS')
set(gca,'FontSize',24)
%%
figure()
x_time = 0.008*(1:size(test_output,2));
plot(x_time,test_output, 'b', 'LineWidth', 2)
hold on
plot(x_time,test_data_true_label(1,end-size(test_output,2)+1:end), 'g', 'LineWidth', 2)
hold on
plot(x_time,0.9*ones(1,length(test_output)), 'r--', 'LineWidth', 2)
hold on
plot(x_time,0.1*ones(1,length(test_output)), 'r--', 'LineWidth', 2)
grid on
xlim([x_time(1) x_time(end)])
ylim([-0.2 1.2])
xlabel('time (sec)')
ylabel('GPS')
legend('Prediction', 'Ground truth', 'GPS=0.9', 'GPS=0.1','Location','north','Orientation','horizontal')
set(gca,'FontSize',24)
%%
figure()
plot([1:1:11900],test_data_sensor')
xlabel('data step')
ylabel('joint angle (deg)')
grid on
set(gca,'FontSize',16)
%% Real-time test
% Max Min 반영하기 (0도:90도 = Minimum:Maximum)
Minimum = load('Minimum.mat');
Minimum = struct2cell(Minimum);
Minimum = Minimum{1,1};
Maximum = load('Maximum.mat');
Maximum = struct2cell(Maximum);
Maximum = Maximum{1,1};

% Load mean/std for input normalization
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\test_data';
filename = 'RG11_Spotting_stat.mat';
filedir = fullfile(pathname,filename);
Spotting_stat = load(filedir);
Spotting_stat = struct2cell(Spotting_stat);
Spotting_stat = Spotting_stat{1,1};

packet_length = 26;
time = 1;
dataset = [];
finger = [];

reservoir_Spotting = [];
reservoir_size_Spotting = 20;
t_Spotting = 1;

fig = figure(1);
while 1
    tmp = fread(s);
    packet_start = find(tmp == 26);
    check = 0;
    for start_point=1:size(packet_start,1)
        check = tmp(packet_start(start_point,:)+(packet_length-1),:)  == 255;
        if check == 1
            break
        end
    end
    
    j = 1;
    count = 0;
    err_check = 0;
    for i=start_point:size(packet_start,1)
        if packet_start(i,:)+(packet_length-1) > size(tmp,1)
            break;
        end
        err_check = (tmp(packet_start(i,:)+(packet_length-1),:) ~= 255);
        if (packet_start(i,:) > 1) && ((tmp(packet_start(i,:)-1,:) ~= 255))
            err_check = err_check + 1;
        end 
        if err_check >= 1
            continue;
        end
        dataset(:,j) = tmp(packet_start(i,:):packet_start(i,:)+(packet_length-1),:);
        for idx=1:10
            finger(idx,j) = 255*dataset(4+2*(idx-1),j) + dataset(4+2*(idx-1)+1,j);
        end
        j = j + 1;
        count = count + 1;
    end
    
    % Min/Max 반영
    for i=1:size(finger,2)
        finger(:,i) = 90*(finger(:,i) - Minimum)./(Maximum - Minimum);
    end
    finger_use = finger(:,1:18);
    
    if size(reservoir_Spotting,2) < reservoir_size_Spotting
        reservoir_Spotting = [reservoir_Spotting,finger_use];
    else
        reservoir_Spotting = [reservoir_Spotting,finger_use];
        input_Spotting = reservoir_Spotting(:,end-reservoir_size_Spotting+1:end);
        
        input_norm_Spotting = (input_Spotting - Spotting_stat(1,:)'*ones(1,size(input_Spotting,2)))./(Spotting_stat(2,:)'*ones(1,size(input_Spotting,2)));
        input_norm_Spotting = input_norm_Spotting';
        
        BiLSTM_hidden1 = BiLSTM_forward(input_norm_Spotting,BiLSTM_weight2,BiLSTM_bias2,'False','concat');

        [Dense_hidden1,~] = Dense_forward(BiLSTM_hidden1(end,:), Dense_weight3, Dense_bias3, 'ReLU');
        [Dense_hidden2,~] = Dense_forward(Dense_hidden1, Dense_weight4, Dense_bias4, 'ReLU');
        [Dense_hidden3,~] = Dense_forward(Dense_hidden2, Dense_weight5, Dense_bias5, 'ReLU');
        [Dense_hidden4,~] = Dense_forward(Dense_hidden3, Dense_weight6, Dense_bias6, 'ReLU');
        [Dense_hidden5,~] = Dense_forward(Dense_hidden4, Dense_weight7, Dense_bias7, 'ReLU');
        [Dense_hidden6,~] = Dense_forward(Dense_hidden5, Dense_weight8, Dense_bias8, 'ReLU');

        [output_Spotting(:,t_Spotting),~] = Dense_forward(Dense_hidden6, Dense_weight9, Dense_bias9, 'sigmoid');
        
        plot(output_Spotting,'k','LineWidth',2)
        hold on
        plot([1,10^8],[0.9,0.9],'r','LineWidth',2)
        hold on
        plot([1,10^8],[0.1,0.1],'r','LineWidth',2)
        
        if t_Spotting < reservoir_size_Spotting + 1
            axis([1,reservoir_size_Spotting,-0.2,1.2])
        else
            axis([t_Spotting-(reservoir_size_Spotting-1),t_Spotting,-0.2,1.2])
        end
        grid on
        title(['Loop time',' ',num2str(t_Spotting)],'FontSize',24)
        hold off
        drawnow
        
        t_Spotting = t_Spotting + 1;
    end
    
%     if time > 200
%         TMCP = finger(1,time+count-200:time+count-1);
%         TIP = finger(2,time+count-200:time+count-1);
%         IMCP = finger(3,time+count-200:time+count-1);
%         IPIP = finger(4,time+count-200:time+count-1);
%         MMCP = finger(5,time+count-200:time+count-1);
%         MPIP = finger(6,time+count-200:time+count-1);
%         RMCP = finger(7,time+count-200:time+count-1);
%         RPIP = finger(8,time+count-200:time+count-1);
%         LMCP = finger(9,time+count-200:time+count-1);
%         LPIP = finger(10,time+count-200:time+count-1);
%         xmin = time+count-1-200+1;
%         xmax = time+count-1;
%         x1 = xmin:1:xmax;
%     else
%         TMCP = finger(1,1:time+count-1);
%         TIP = finger(2,1:time+count-1);
%         IMCP = finger(3,1:time+count-1);
%         IPIP = finger(4,1:time+count-1);
%         MMCP = finger(5,1:time+count-1);
%         MPIP = finger(6,1:time+count-1);
%         RMCP = finger(7,1:time+count-1);
%         RPIP = finger(8,1:time+count-1);
%         LMCP = finger(9,1:time+count-1);
%         LPIP = finger(10,1:time+count-1);
%         xmin = 1;
%         xmax = 200;
%         x1 = 1:1:time+count-1;
%     end
%     figure(1)
%     cla
%     plot(x1,TMCP)
%     hold on
%     plot(x1,TIP)
%     hold on
%     plot(x1,IMCP)
%     hold on
%     plot(x1,IPIP)
%     hold on
%     plot(x1,MMCP)
%     hold on
%     plot(x1,MPIP)
%     hold on
%     plot(x1,RMCP)
%     hold on
%     plot(x1,RPIP)
%     hold on
%     plot(x1,LMCP)
%     hold on
%     plot(x1,LPIP)    
%     ylim([500 3000])
%     xlim([xmin, xmax])
%     grid on
%     drawnow
%     
%     time = time + count;
end