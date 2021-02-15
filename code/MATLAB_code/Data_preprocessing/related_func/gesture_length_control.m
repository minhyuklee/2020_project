function output = gesture_length_control(input)
% input: ���� �� * 1 * ����ó Ŭ����
% output: ���� �� * 1 * ����ó Ŭ����
%% Velocity ��� �� ����ó ���� ô�� ���
% Velocity ���
% low-pass filter
Fs = 100; % sampling freq
n = 2; % ����
Wn = 4; % cut off freq
Fn = Fs/2; % Nyquist freq
ftype = 'low';
[b,a] = butter(n,Wn/Fn, ftype);
velocity_training = {};
for i=1:size(input,1)
    for j=1:size(input,3)
        for k=2:size(input{i,1,j},2)
            velocity_training{i,1,j}(:,k-1) = norm(input{i,1,j}(:,k) - input{i,1,j}(:,k-1));
        end
        velocity_training{i,1,j} = filter(b,a,velocity_training{i,1,j});
        tmp_offset = (mean(velocity_training{i,1,j}(:,1:20),2) + mean(velocity_training{i,1,j}(:,end-19:end),2))/2;
        velocity_training{i,1,j} = velocity_training{i,1,j}-tmp_offset*ones(1,size(velocity_training{i,1,j},2));
        velocity_training{i,1,j} = abs(velocity_training{i,1,j});
        % velocity ������. (����ó �ܰ� ���� ��Ȯ�� �ϱ� ����)
        velocity_training{i,1,j} = velocity_training{i,1,j}.*velocity_training{i,1,j};
        velocity_training{i,1,j}(1,1:10) = 0;
    end
end

% ����ó ���� ô�� ���
GPS = {}; % input�� ���� ������.
for i=1:size(velocity_training,1)
    for j=1:size(velocity_training,3)
        integrat_sum = 0;
        integrat = integrat_sum;
        for k=1:size(velocity_training{i,1,j},2)
            integrat_sum = integrat_sum + velocity_training{i,1,j}(:,k);
            integrat = [integrat,integrat_sum];
        end
        integrat = 1/integrat(end)*integrat;
        GPS{i,1,j} = integrat;
    end
end
%% Velocity, GPS �ö�
g_num = 1;
figure()
for i=1:size(GPS,1)
    plot(GPS{i,1,g_num})
    hold on
end
grid on

figure()
for i=1:size(velocity_training,1)
    plot(velocity_training{i,1,g_num})
    hold on
end
grid on
%% ����ó �ܰ� �� ���� ����
phase_length_org = [];
feature_point = [];
% stage_boundary = [0.01,0.99];
for i=1:size(input,1)
    for j=1:size(input,3)
        feature_point(i,1,j) = 1;
%         for k=1:size(GPS{i,1,j},2)
%             if GPS{i,1,j}(1,k) > stage_boundary(1,1)
%                 feature_point(i,2,j) = k-1;
%                 break
%             end
%         end        
%         for k=1:size(GPS{i,1,j},2)
%             if GPS{i,1,j}(1,k) > stage_boundary(1,2)
%                 feature_point(i,3,j) = k-1;
%                 break
%             end
%         end
        for k=1:size(velocity_training{i,1,j},2)
            if velocity_training{i,1,j}(1,k) > 0.35 % 0.35
                feature_point(i,2,j) = k-1;
                break
            end
        end        
        for k=size(velocity_training{i,1,j},2):-1:1
            if velocity_training{i,1,j}(1,k) > 0.35 % 0.35
                feature_point(i,3,j) = k-1;
                break
            end
        end
        feature_point(i,4,j) = size(GPS{i,1,j},2);

        phase_length_org(i,1,j) = feature_point(i,2,j) - feature_point(i,1,j) + 1;
        phase_length_org(i,2,j) = feature_point(i,3,j) - feature_point(i,2,j);
        phase_length_org(i,3,j) = feature_point(i,4,j) - feature_point(i,3,j);
    end
end

phase_length_cont = [];
for i=1:size(phase_length_org,1)
    for j=1:size(phase_length_org,3)
        % �ʱ� ���̸� �߱�� �����ϰ� ���߱�
        if phase_length_org(i,1,j) < phase_length_org(i,2,j) % �ʱ� ���̰� �߱� ���̺��� ª�� ���
            phase_length_cont(i,1,j) = phase_length_org(i,1,j);
        else
            % �߱� ���̰� �ʱ� ���̺��� ª�� ���
            phase_length_cont(i,1,j) = phase_length_org(i,2,j);
        end
        
        % �߱� ���̴� �״��
        phase_length_cont(i,2,j) = phase_length_org(i,2,j);
        
        % �ı� ���̸� �߱�� �����ϰ� ���߱�
        if phase_length_org(i,3,j) < phase_length_org(i,2,j) % �ı� ���̰� �߱� ���̺��� ª�� ���
            phase_length_cont(i,3,j) = phase_length_org(i,3,j);
        else
            % �߱� ���̰� �ı� ���̺��� ª�� ���
            phase_length_cont(i,3,j) = phase_length_org(i,2,j);
        end
    end
end

output = {};
for i=1:size(input,1)
    for j=1:size(input,3)
        output{i,1,j} = input{i,1,j}(:,feature_point(i,2,j)-phase_length_cont(i,1,j)+1:feature_point(i,3,j)+phase_length_cont(i,3,j));
%         output{i,1,j} = input{i,1,j};
    end
end
end