clc
clear all
close all
%% Bluetooth Connect
delete(instrfindall) % 연결되어있는 시리얼포트 정보 지우기
s = serial('com5','BaudRate',115200,'DataBits',8,'Parity','non','StopBits',1,'FlowControl','non','Terminator','CR');
fopen(s);
%% Max Min 반영하기 (0도:90도 = Minimum:Maximum)
Minimum = load('Minimum.mat');
Minimum = struct2cell(Minimum);
Minimum = Minimum{1,1};
Maximum = load('Maximum.mat');
Maximum = struct2cell(Maximum);
Maximum = Maximum{1,1};
%% 연결된 제스처 데이터 생성
packet_length = 26;
time = 1;
dataset = [];
datalength = 1;
finger = [];
mark = [];

% GUI button variables initialization
% GUI버튼과 관련된 변수 초기화
store_mark = 0;
stop = 0;
toggle = 0;
mark = [];

% button
button_test_sequence_generation

while stop == 0
    tmp = fread(s);
    packet_start = find(tmp == 26);
    check = 0;
    for start_point=1:size(packet_start,1)
        check = tmp(packet_start(start_point,:)+(packet_length-1),:)  == 255;
        if check == 1
            break
        end
    end

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
        dataset(:,datalength) = tmp(packet_start(i,:):packet_start(i,:)+(packet_length-1),:);
        for idx=1:10
            finger(idx,datalength) = 255*dataset(4+2*(idx-1),datalength) + dataset(4+2*(idx-1)+1,datalength);
            finger(idx,datalength) = 90*(finger(idx,datalength)-Minimum(idx))/(Maximum(idx)-Minimum(idx));
        end
        if isempty(find(abs(finger)>180,1)) ~= 1
            disp('관절각도가 180도 이상입니다.')
        end        
        mark(1,datalength) = store_mark*1000; % 시퀀스 저장 순간 결정
        mark(2,datalength) = toggle*1000; % 제스처 구간 결정
        
        datalength = datalength + 1;
        count = count + 1;
    end
    
    if time > 200
        TMCP = finger(1,time+count-200:time+count-1);
        TIP = finger(2,time+count-200:time+count-1);
        IMCP = finger(3,time+count-200:time+count-1);
        IPIP = finger(4,time+count-200:time+count-1);
        MMCP = finger(5,time+count-200:time+count-1);
        MPIP = finger(6,time+count-200:time+count-1);
        RMCP = finger(7,time+count-200:time+count-1);
        RPIP = finger(8,time+count-200:time+count-1);
        LMCP = finger(9,time+count-200:time+count-1);
        LPIP = finger(10,time+count-200:time+count-1);
        xmin = time+count-1-200+1;
        xmax = time+count-1;
        x1 = xmin:1:xmax;
    else
        TMCP = finger(1,1:time+count-1);
        TIP = finger(2,1:time+count-1);
        IMCP = finger(3,1:time+count-1);
        IPIP = finger(4,1:time+count-1);
        MMCP = finger(5,1:time+count-1);
        MPIP = finger(6,1:time+count-1);
        RMCP = finger(7,1:time+count-1);
        RPIP = finger(8,1:time+count-1);
        LMCP = finger(9,1:time+count-1);
        LPIP = finger(10,1:time+count-1);
        xmin = 1;
        xmax = 200;
        x1 = 1:1:time+count-1;
    end
    figure(1)
    cla
    plot(x1,TMCP)
    hold on
    plot(x1,TIP)
    hold on
    plot(x1,IMCP)
    hold on
    plot(x1,IPIP)
    hold on
    plot(x1,MMCP)
    hold on
    plot(x1,MPIP)
    hold on
    plot(x1,RMCP)
    hold on
    plot(x1,RPIP)
    hold on
    plot(x1,LMCP)
    hold on
    plot(x1,LPIP)    
    ylim([-90 180])
    xlim([xmin, xmax])
    grid on
    drawnow
    
    time = time + count;
end
% for i=1:10
%     finger(i,:) = filter(b,a,finger(i,:));
% end
%%
figure()
plot(finger')
gesture = finger(:,1100:1900);
%% 테스트 시퀀스 저장
sv_seq = find(mark(1,:)==1000);
gesture = [finger(:,sv_seq);1/1000*mark(2,sv_seq)];
%% plot
figure()
for i=1:5
    plot(gesture{i,1}(4,:))
    hold on
end
%% Save data and finish
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data generation\Data Gathering\complete_set';
filename = 'raw_Rgesture_series_test.mat';
filedir = fullfile(pathname,filename);
save(filedir,'gesture')
disp('데이터 저장이 끝났습니다.')