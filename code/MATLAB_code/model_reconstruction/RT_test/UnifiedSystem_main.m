% Real-time dynamic hand gesture recognition algorithm
% Main function
clc
clear all
close all
%% 모델 로드
% Gesture spotting 모델
Spotting = load('G11_Spotting.mat'); % G11_Spotting_complete3.mat
Spotting = struct2cell(Spotting);
Spotting = Spotting{1,1};

Spotting_stat = load('G11_Spotting_stat.mat'); % G11_Spotting_stat_complete3
Spotting_stat = struct2cell(Spotting_stat);
Spotting_stat = Spotting_stat{1,1};

% Gesture recognition 모델 로드
Recognition = load('G11_Recognition.mat');
Recognition = struct2cell(Recognition);
Recognition = Recognition{1,1};

Recognition_stat = load('G11_Recognition_stat.mat');
Recognition_stat = struct2cell(Recognition_stat);
Recognition_stat = Recognition_stat{1,1};
%% OFF LINE TEST: 테스트 시퀀스 로드
test_sequence = load('SequentialGesture8_G11.mat');
test_sequence = struct2cell(test_sequence);
test_sequence = test_sequence{1,1};
%% 테스트 데이터에 학습 모델 적용
reservoir_Spotting = [];
reservoir_Recognition = [];
reservoir_size_Spotting = 50;
GPS_thres_early = 0.1;
GPS_thres_end = 0.9;
tolerance_Spotting = 0.1;
tolerance_Recognition = 2;
output_Spotting = zeros(1,size(test_sequence,2));
output_Recognition = zeros(11,size(test_sequence,2));

for t=1:size(test_sequence,2) % Spotting목적의 reservoir는 reservoir_size마다 새로 reset
    % Gesture spotting
    if size(reservoir_Spotting,2) < reservoir_size_Spotting
        reservoir_Spotting = [reservoir_Spotting,test_sequence(1:10,t)];
    else
        input_Spotting = reservoir_Spotting;
        input_compressed_Spotting = seq_compress_v2(input_Spotting,tolerance_Spotting);
        input_norm_Spotting = (input_compressed_Spotting - Spotting_stat(:,1)*ones(1,size(input_compressed_Spotting,2)))./(Spotting_stat(:,2)*ones(1,size(input_compressed_Spotting,2)));
        
        [hidden1_LSTM_Spotting,~] = lstm_forward_test(input_norm_Spotting,Spotting{1,1},Spotting{2,1});
        [hidden2_LSTM_Spotting,~] = lstm_forward_test(hidden1_LSTM_Spotting,Spotting{1,2},Spotting{2,2});
        concat = [hidden2_LSTM_Spotting(:,end);input_norm_Spotting(:,end)];
        [hidden1_ANN_Recognition,~] = ANN_forward(concat,Spotting{1,3},Spotting{2,3},'ReLU');
        [hidden2_ANN_Recognition,~] = ANN_forward(hidden1_ANN_Recognition,Spotting{1,4},Spotting{2,4},'ReLU');
        [hidden3_ANN_Recognition,~] = ANN_forward(hidden2_ANN_Recognition,Spotting{1,5},Spotting{2,5},'ReLU');
        [hidden4_ANN,~] = ANN_forward(hidden3_ANN_Recognition,Spotting{1,6},Spotting{2,6},'ReLU');
        [hidden5_ANN,~] = ANN_forward(hidden4_ANN,Spotting{1,7},Spotting{2,7},'ReLU');
        [hidden6_ANN,~] = ANN_forward(hidden5_ANN,Spotting{1,8},Spotting{2,8},'ReLU');
    
        output_Spotting(:,t) = output_forward_test(hidden6_ANN,Spotting{1,9},Spotting{2,9},'sigmoid');
        
        if output_Spotting(:,t) < GPS_thres_early
            flag_reservoir = 0;
        end
        if output_Spotting(:,t) > GPS_thres_end
            flag_reservoir = 1;
        end
        reservoir_Spotting = [];
    end
    
    if t > reservoir_size_Spotting
        if flag_reservoir == 0
            reservoir_Recognition = [reservoir_Recognition,test_sequence(1:10,t)];
            flag_reservoir_old = flag_reservoir;
            continue;
        end

        if (flag_reservoir - flag_reservoir_old) == 1
            input_Recognition = reservoir_Recognition;
            input_compressed_Recognition = seq_compress_v2(input_Recognition,tolerance_Recognition);
            input_norm_Recognition = (input_compressed_Recognition - Recognition_stat(:,1)*ones(1,size(input_compressed_Recognition,2)))./(Recognition_stat(:,2)*ones(1,size(input_compressed_Recognition,2)));

            [hidden1_LSTM_Recognition,~] = lstm_forward_test(input_norm_Recognition,Recognition{1,1},Recognition{2,1});
            [hidden2_LSTM_Recognition,~] = lstm_forward_test(hidden1_LSTM_Recognition,Recognition{1,2},Recognition{2,2});

            [hidden1_ANN_Recognition,~] = ANN_forward(hidden2_LSTM_Recognition(:,end),Recognition{1,3},Recognition{2,3},'ReLU');
            [hidden2_ANN_Recognition,~] = ANN_forward(hidden1_ANN_Recognition,Recognition{1,4},Recognition{2,4},'ReLU');
            [hidden3_ANN_Recognition,~] = ANN_forward(hidden2_ANN_Recognition,Recognition{1,5},Recognition{2,5},'ReLU');         
            output_Recognition(:,t) = output_forward_test(hidden3_ANN_Recognition,Recognition{1,6},Recognition{2,6},'many2one');

            reservoir_Recognition = [];
            flag_reservoir_old = flag_reservoir;
        end
    end
end
%%
figure(1)
plot(find(output_Spotting~=0),output_Spotting(:,find(output_Spotting~=0)),'r')
hold on
plot(test_sequence(11,:))
ylim([-0.2 1.2])
grid on
%%
tt = find(sum(output_Recognition,1)~=0);
figure(2)
for i=1:size(tt,2)
    bar(output_Recognition(:,tt(i)))
    ylim([0 1])
    ylabel('Probability')
    title('Test')
    grid on
    drawnow
    pause(1)
end
%% ON LINE TEST
% Max Min 반영하기 (0도:90도 = Minimum:Maximum)
Minimum = load('Minimum.mat');
Minimum = struct2cell(Minimum);
Minimum = Minimum{1,1};
Maximum = load('Maximum.mat');
Maximum = struct2cell(Maximum);
Maximum = Maximum{1,1};

% Image load
Image_filename = {'G1_Best','G2_Just Kidding','G3_Like','G4_Nobody','G5_No','G6_Pinch',...
                    'G7_Sister','G8_Understand','G9_Want','G10_Who','G11_Why'};
for i=1:size(Image_filename,2)
    Image_filename{1,i} = strcat(Image_filename{1,i},'.png');
end
Directory = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\model_reconstruction\11개 제스처 사진\';
Image_variable = {};
for i=1:size(Image_filename,2)
    Image_variable{i,1} = imread(strcat(Directory,Image_filename{1,i}));
end

packet_length = 61;
time = 1;
BT_data = [];
finger = [];
finger_use = zeros(10,1);

reservoir_Spotting = [];
reservoir_Recognition = [];
reservoir_size_Spotting = 35; % 35
GPS_thres_early = 0.1;
GPS_thres_end = 0.9;
% tolerance_Spotting = 0.1;
tolerance_Recognition = 0.5;
t = 0;
t_Spotting = 1;
t_Recognition = 1;
step1_time = [];

flag_second_phase = 0;
flag_storage_Recognition = 0;
flag_third_phase = 0;
flag_true_third_phase_check = 0;
clear output_Spotting output_Recognition
fig = figure(1);
fig.PaperPositionMode = 'auto';
set(fig,'Position',[0 0 1500 800])
while 1
    tmp = fread(s);
    packet_start = find(tmp == 64);
    check = 0;
    for start_point=1:size(packet_start,1)
        check = tmp(packet_start(start_point,:)+(packet_length-1),:)  == 10;
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
        err_check = (tmp(packet_start(i,:)+(packet_length-1),:) ~= 10);
        if (packet_start(i,:) > 1) && ((tmp(packet_start(i,:)-1,:) ~= 10))
            err_check = err_check + 1;
        end 
        if err_check >= 1
            continue;
        end
        BT_data(:,j) = tmp(packet_start(i,:):packet_start(i,:)+(packet_length-1),:);
        for idx=1:10
            finger(idx,j) = 10^3*str2double(char(BT_data(19+4*(idx-1),j))) + 10^2*str2double(char(BT_data(19+4*(idx-1)+1,j))) + 10^1*str2double(char(BT_data(19+4*(idx-1)+2,j))) + str2double(char(BT_data(19+4*(idx-1)+3,j)));
        end
        j = j + 1;
        count = count + 1;
    end
    
    % Min/Max 반영
    for i=1:size(finger,2)
        finger(:,i) = 90*(finger(:,i) - Minimum)./(Maximum - Minimum);
    end  
    finger_use = finger(:,1:7);
    t = t + 7;
    
    % 1단계: Gesture spotting
    if size(reservoir_Spotting,2) < reservoir_size_Spotting
        flag_second_phase = 0.5;
        reservoir_Spotting = [reservoir_Spotting,finger_use];
    else
        tic
        % reservoir_Spotting 저장방식 변경으로 추가된 사항
        reservoir_Spotting = [reservoir_Spotting,finger_use];
        %
        input_Spotting = reservoir_Spotting(:,end-reservoir_size_Spotting+1:end); % reservoir_Spotting 저장방식 변경으로 (:,end-reservoir_size_Spotting+1:end)로 변경됨. 원래는 그냥 reservoir_Spotting
%         input_compressed_Spotting = seq_compress_v2(input_Spotting,tolerance_Spotting);
        input_compressed_Spotting = input_Spotting;

        input_norm_Spotting = (input_compressed_Spotting - Spotting_stat(:,1)*ones(1,size(input_compressed_Spotting,2)))./(Spotting_stat(:,2)*ones(1,size(input_compressed_Spotting,2)));
        
        [hidden1_LSTM_Spotting,~] = lstm_forward_test(input_norm_Spotting,Spotting{1,1},Spotting{2,1});
        [hidden2_LSTM_Spotting,~] = lstm_forward_test(hidden1_LSTM_Spotting,Spotting{1,2},Spotting{2,2});
        concat = [hidden2_LSTM_Spotting(:,end);input_norm_Spotting(:,end)];
        [hidden1_ANN_Recognition,~] = ANN_forward(concat,Spotting{1,3},Spotting{2,3},'ReLU');
        [hidden2_ANN_Recognition,~] = ANN_forward(hidden1_ANN_Recognition,Spotting{1,4},Spotting{2,4},'ReLU');
        [hidden3_ANN_Recognition,~] = ANN_forward(hidden2_ANN_Recognition,Spotting{1,5},Spotting{2,5},'ReLU');
        [hidden4_ANN,~] = ANN_forward(hidden3_ANN_Recognition,Spotting{1,6},Spotting{2,6},'ReLU');
        [hidden5_ANN,~] = ANN_forward(hidden4_ANN,Spotting{1,7},Spotting{2,7},'ReLU');
        [hidden6_ANN,~] = ANN_forward(hidden5_ANN,Spotting{1,8},Spotting{2,8},'ReLU');
    
        output_Spotting(:,t_Spotting) = output_forward_test(hidden6_ANN,Spotting{1,9},Spotting{2,9},'sigmoid');
        step1_time = [step1_time,toc];
        subplot(1,2,1)
        plot(output_Spotting,'k','LineWidth',8)
        hold on
        plot([1,10^8],[0.9,0.9],'r','LineWidth',8)
        hold on
        plot([1,10^8],[0.1,0.1],'r','LineWidth',8)
        
        if t_Spotting < reservoir_size_Spotting + 1
            axis([1,reservoir_size_Spotting,-0.1,1.2])
        else
            axis([t_Spotting-(reservoir_size_Spotting-1),t_Spotting,-0.1,1.2])
        end
        grid on
        title(['Loop time','  ',num2str(t_Spotting)],'FontSize',24)
        hold off
        drawnow
        
        t_Spotting = t_Spotting + 1;
        
        if output_Spotting(:,end) < GPS_thres_early
            flag_second_phase = 1;
        end
        if output_Spotting(:,end) > GPS_thres_end
            flag_storage_Recognition = 0;
            flag_second_phase = 0;
        end
        
        if flag_storage_Recognition == 1
            flag_second_phase = 0.5; % 제스처 진행중
        end
        
        % reservoir_Spotting 저장방식 변경으로 변화된 부분
%         reservoir_Spotting = []; % 기존: []
    end

    % 2단계: reservoir_recognition에 데이터 저장
    if flag_second_phase == 1 % 제스처 시작
        reservoir_Recognition = [];
        flag_storage_Recognition = 1;
        flag_third_phase = 0;
    end
    if flag_storage_Recognition == 1 % 제스처 진행중
        reservoir_Recognition = [reservoir_Recognition,finger_use];
        flag_third_phase = 0;
    end
    if flag_second_phase == 0 % 제스처 끝
        input_Recognition = reservoir_Recognition;
        reservoir_Recognition = [];
        flag_storage_Recognition = 0;
        flag_true_third_phase_check = 1;
    end
    
    if flag_true_third_phase_check == 1
        if size(input_Recognition,2) == 0
            flag_third_phase = 0;
        else
            flag_third_phase = 1;
        end
    end
    flag_true_third_phase_check = 0;
    
    % 3단계: Gesture recognition  
    if flag_third_phase == 1
%         size(input_Recognition,2)
        input_compressed_Recognition = seq_compress_v2(input_Recognition,tolerance_Recognition);
%         size(input_compressed_Recognition,2)
        input_norm_Recognition = (input_compressed_Recognition - Recognition_stat(:,1)*ones(1,size(input_compressed_Recognition,2)))./(Recognition_stat(:,2)*ones(1,size(input_compressed_Recognition,2)));

        [hidden1_LSTM_Recognition,~] = lstm_forward_test(input_norm_Recognition,Recognition{1,1},Recognition{2,1});
        [hidden2_LSTM_Recognition,~] = lstm_forward_test(hidden1_LSTM_Recognition,Recognition{1,2},Recognition{2,2});
   
        [hidden1_ANN_Recognition,~] = ANN_forward(hidden2_LSTM_Recognition(:,end),Recognition{1,3},Recognition{2,3},'ReLU');
        [hidden2_ANN_Recognition,~] = ANN_forward(hidden1_ANN_Recognition,Recognition{1,4},Recognition{2,4},'ReLU');
        [hidden3_ANN_Recognition,~] = ANN_forward(hidden2_ANN_Recognition,Recognition{1,5},Recognition{2,5},'ReLU');         
        
        output_Recognition(:,t_Recognition) = output_forward_test(hidden3_ANN_Recognition,Recognition{1,6},Recognition{2,6},'many2one');

        subplot(1,2,2)
%         bar(output_Recognition(:,end))
%         ylim([0 1])
%         ylabel('Probability')
%         title('Test')
%         grid on
        [~,whichone] = max(output_Recognition(:,end));
        imshow(Image_variable{whichone,1})
        drawnow
        
        t_Recognition = t_Recognition + 1;
        
        flag_third_phase = 0;
    end

%     figure(2)
%     plot(1:1:count,finger(1,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(2,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(3,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(4,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(5,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(6,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(7,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(8,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(9,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(10,time:time+count-1))    
%     ylim([-120 120])
%     xlim([1 7])
%     grid on
%     drawnow
end