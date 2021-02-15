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
%% Obtain train data
% Train data 생성을 위한 data receive

packet_length = 26;

global cnt Gesture_ind
cnt = 0; Gesture_ind = 0;
finger = [];
dataset = [];
mark = [];
time = 1;
datalength = 1;

% GUI button variables initialization
stop = 0;
toggle = 0;
cancel = 0;
gesture_select = 0;

% button
button_train

while stop == 0
    tic
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
        mark(1,datalength) = toggle*1000;
        mark(2,datalength) = cancel*1000;
        mark(3,datalength) = Gesture_ind;
        
        cancel = 0;

        datalength = datalength + 1;
        count = count + 1;
    end
    sampling = toc;
    
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
disp('실험이 완료되었습니다.')
%% Cancellation
% 변수 mark 이용하여 제스처 시작, 끝 위치 파악
gesture_start = [];
gesture_end = [];
for i=2:size(mark,2)
    if (mark(1,i-1) == 0) && (mark(1,i) == 1000)
        gesture_start = [gesture_start i];
    end
    if (mark(1,i-1) == 1000) && (mark(1,i) == 0)
        gesture_end = [gesture_end i];
    end    
end
gesture_collect = [gesture_start;gesture_end];

cancel_gesture = [];
for i=2:size(mark,2)
    if (mark(2,i-1) == 0) && (mark(2,i) == 1000)
        cancel_gesture = [cancel_gesture i];
    end
end

count_cancel = length(cancel_gesture);
erase_idx = [];
if count_cancel ~= 0
    for i=1:size(gesture_collect,2)-1
        for j=1:size(cancel_gesture,2)
            if (gesture_collect(2,i) < cancel_gesture(j)) && (gesture_collect(1,i+1) > cancel_gesture(j))
                erase_idx = [erase_idx i];
            end
        end
    end
end
gesture_collect(:,erase_idx) = [];
disp('수집된 데이터에서 제거해야 할 항목이 제거되었습니다.')

% 셀 gesture에 데이터 및 라벨 기록
gesture = cell(1,2);
label_mat = eye(17);
for i=1:size(gesture_collect,2)
    gesture{i,1} = finger(1:10,gesture_collect(1,i):gesture_collect(2,i));
    gesture{i,2} = label_mat(:,mark(3,gesture_collect(1,i)));
end
disp('데이터가 클래스 별로 분류되었습니다.')
% plot
% figure()
% for i=1:5
%     plot(gesture{i,1}(4,:))
%     hold on
% end
%% Save data and finish
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data generation\Data Gathering\data';
filename = '20200831KKY1_RG17_R3.mat';
filedir = fullfile(pathname,filename);
save(filedir,'gesture')
disp('데이터 저장이 끝났습니다.')