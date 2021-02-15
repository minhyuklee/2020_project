function output = GPS_trainset_generation(input)
% input: ���� �� * 1 * ����ó Ŭ����
% output: ���� �� * 2 * ����ó Ŭ����
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
%         velocity_training{i,1,j} = velocity_training{i,1,j}.*velocity_training{i,1,j};
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

output = {};
for i=1:size(input,1)
    for j=1:size(input,3)
        output{i,1,j} = input{i,1,j};
        output{i,2,j} = GPS{i,1,j};
    end
end
%% Velocity, GPS �ö�
g_num = 1;
figure()
for i=1:size(GPS,1)
    plot(GPS{i,1,g_num},'LineWidth',2)
    hold on
end
xlabel('time step')
ylabel('GPS')
set(gca,'FontSize',16)
grid on

figure()
for i=1:size(velocity_training,1)
    plot(velocity_training{i,1,g_num},'LineWidth',2)
    hold on
end
xlabel('time step')
ylabel('absolute velocity')
set(gca,'FontSize',16)
grid on
end