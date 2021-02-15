%% 센서 신호 Min/Max 측정용
clc
clear all
close all
%% Bluetooth Connect
delete(instrfindall) % 연결되어있는 시리얼포트 정보 지우기
s = serial('com5','BaudRate',115200,'DataBits',8,'Parity','non','StopBits',1,'FlowControl','non','Terminator','CR');
fopen(s);
%% Min. & Max. 값을 얻기 위한 데이터 receive
packet_length = 61;
time = 1;
dataset = [];
datalength = 1;
finger = [];
mark = [];

% low-pass filter
Fs = 200; % sampling freq
n = 6; % 차수
Wn = 40; % cut off freq
Fn = Fs/2; % Nyquist freq
ftype = 'low';
[b,a] = butter(n,Wn/Fn, ftype);

% GUI button variables initialization
% GUI버튼과 관련된 변수 초기화
stop = 0;
min_point = 0;
max_point = 0;

% button
button_MinMax

while stop == 0
    tic
    tmp = fread(s);
    packet_start = find(tmp == 64);
    check = 0;
    for start_point=1:size(packet_start,1)
        check = tmp(packet_start(start_point,:)+(packet_length-1),:)  == 10;
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
        err_check = (tmp(packet_start(i,:)+(packet_length-1),:) ~= 10);
        if (packet_start(i,:) > 1) && ((tmp(packet_start(i,:)-1,:) ~= 10))
            err_check = err_check + 1;
        end 
        if err_check >= 1
            continue;
        end
        dataset(:,datalength) = tmp(packet_start(i,:):packet_start(i,:)+(packet_length-1),:);
        for idx=1:10
            finger(idx,datalength) = 10^3*str2double(char(dataset(19+4*(idx-1),datalength))) + 10^2*str2double(char(dataset(19+4*(idx-1)+1,datalength))) + 10^1*str2double(char(dataset(19+4*(idx-1)+2,datalength))) + str2double(char(dataset(19+4*(idx-1)+3,datalength)));
        end
        mark(1,datalength) = min_point*1000; % Minimum 일때 표시
        mark(2,datalength) = max_point*1000; % Maximum 일때 표시
        
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
    ylim([500 3000])
    xlim([xmin, xmax])
    grid on
    drawnow
    
    
    time = time + count;
end
% for i=1:10
%     finger(i,:) = filter(b,a,finger(i,:));
% end

% Obatin Min & Max values
j = 1; r = 1;
Min_norm = [ ]; % Min. 값의 요소가 될 data
Max_norm = [ ]; % Max. 값의 요소가 될 data
for i = 1:size(finger,2)
    if mark(1,i) == 1000
       Min_norm(:,j) = finger(:,i);
       j = j+1;
    end
    if mark(2,i) == 1000
       Max_norm(:,r) = finger(:,i);
       r = r+1;
    end
end
Minimum = mean(Min_norm,2); % 각 센서별 최솟값, 최댓값 구함
Maximum = mean(Max_norm,2); 
save('Minimum','Minimum')
save('Maximum','Maximum')